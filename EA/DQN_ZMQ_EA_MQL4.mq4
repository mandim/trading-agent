//+------------------------------------------------------------------+
//|                                              DQN_ZMQ_EA_MQL4.mq4 |
//| Live DQN trading EA for MetaTrader 4 using a ZeroMQ Python server |
//| Requires libzmq.dll in MQL4/Libraries and DLL imports enabled     |
//+------------------------------------------------------------------+
#property strict

// ==================== User Inputs ====================
extern string Endpoint = "tcp://127.0.0.1:6000"; // must match serve_dqn.py --bind
extern int WindowLen = 32;                       // must match server
extern int RawWindowLen = 300;                   // Higher history reduces indicator drift (server needs >= 60 for std_50)
extern int StepMs = 500;                         // query interval (ms)
extern bool UseTickAskBid = true;                // use MarketInfo(Symbol(),MODE_ASK/BID)
extern int SlPips = 40;                          // SL in pips
extern int TpPips = 50;                          // TP in pips
extern int RiskPercent = 1;                      // Risk percentage for each position
extern bool OnePosition = true;                  // close opposite before open
extern int SlippagePoints = 0;                   // max slippage (points) - set to 0 for parity
extern int Magic = 987654;                       // magic number for EA orders
extern int MinHoldBars = 5;                      // must match env.min_hold_bars
extern bool AllowReverse = false;                // must match env.allow_reverse (recommended false)

// ==================== Parity / Validation Mode ====================
// When ParityMode=true, the EA should behave 1:1 with the Python training env (TradingEnv).
extern bool ParityMode = true;
extern double ParityLot = 1.0;               // training env lot=1.0
extern int ParityMinHoldBars = 5;            // training env min_hold_bars
extern int ParityCooldownBarsAfterClose = 3; // training env cooldown_bars_after_close
extern bool ParityAllowReverse = false;      // training env allow_reverse
extern bool ParityDisableDailyReset = true;  // disable EA daily reset for strict parity

// ==================== Tick History (for server-side normalization) ====================
extern bool SendTickHistory = true;          // send recent tick ask/bid history to server
extern int MaxTickHistory = 10000;           // cap to avoid oversized ZMQ payloads

// --- feature config ---
#define MAX_FEATURES 5 // raw OHLVC

double BarWin[][MAX_FEATURES]; // [WindowLen][MAX_FEATURES]

// ---- ZMQ state machine for REQ socket ----
uint gLastSendMs = 0;

int gCooldownBars = 0; // cooldown after close (match env)

// --- Parity helpers ---
int GetMinHoldBars() { return (ParityMode ? ParityMinHoldBars : MinHoldBars); }
int GetCooldownBars() { return (ParityMode ? ParityCooldownBarsAfterClose : 3); } // default 3 kept for backward compat
bool GetAllowReverse() { return (ParityMode ? ParityAllowReverse : AllowReverse); }
double GetLotParity() { return (ParityMode ? ParityLot : -1.0); }

// Track position transitions so cooldown also applies to TP/SL closes (env parity)
int gPrevDir = 0;
int gPrevTicket = -1;
datetime gPrevEntryTime = 0;

double gEntryAsk = 0.0;
double gEntryBid = 0.0;
bool gHasEntry = false;

// ================= TICK HISTORY =================
double gTickAsk[]; // rolling tick ask history since last request
double gTickBid[]; // rolling tick bid history since last request

// ================= BAR GATING =================
datetime g_last_bar_time = 0;

// ================= CSV LOGGING =================
int g_fh_actions = INVALID_HANDLE;
extern bool LogActions = true;
extern string ActionsFile = "ea_actions_2025.csv";
int g_fh_exec = INVALID_HANDLE;
extern bool LogExec = true;
extern string ExecFile = "ea_exec_log.csv";

// ==================== ZeroMQ Imports ====================
#define ZMQ_REQ 3
#define ZMQ_DONTWAIT 1

#import "libzmq.dll"
int zmq_ctx_new(); // void* to int (32-bit handle)
int zmq_ctx_term(int ctx);
int zmq_socket(int ctx, int type);
int zmq_close(int s);
int zmq_connect(int s, uchar &endpoint[]);
int zmq_send(int s, uchar &data[], int size, int flags);
int zmq_recv(int s, uchar &buf[], int size, int flags);
int zmq_errno();
#import

// ==================== ZMQ Client (inline) ====================
class CZmqClient
{
private:
  int m_ctx;
  int m_sock;
  int m_recv_cap;

public:
  CZmqClient() : m_ctx(0), m_sock(0), m_recv_cap(262144) {}
  bool Init(const string endpoint)
  {
    if (m_ctx != 0 || m_sock != 0)
      Shutdown();

    m_ctx = zmq_ctx_new();
    if (m_ctx == 0)
    {
      Print("[DQN] zmq_ctx_new failed");
      return false;
    }

    m_sock = zmq_socket(m_ctx, ZMQ_REQ);
    if (m_sock == 0)
    {
      int err = zmq_errno();
      Print("[DQN] zmq_socket failed, errno=", err);
      zmq_ctx_term(m_ctx);
      m_ctx = 0;
      return false;
    }

    // Build narrow (ANSI) C-string endpoint
    int len = StringLen(endpoint);
    uchar e[];
    ArrayResize(e, len + 2); // +1 for '\0', +1 safety
    int written = StringToCharArray(endpoint, e, 0, WHOLE_ARRAY, CP_ACP);
    if (written <= 0)
    {
      Print("[DQN] StringToCharArray failed for endpoint: ", endpoint);
      Shutdown();
      return false;
    }
    e[written] = 0; // explicit C-string terminator

    int rc = zmq_connect(m_sock, e);
    if (rc != 0)
    {
      int err = zmq_errno();
      Print("[DQN] zmq_connect failed, errno=", err, " endpoint=", endpoint);
      Shutdown();
      return false;
    }

    Print("[DQN] zmq_connect OK to ", endpoint);
    return true;
  }

  bool Send(const string text)
  {
    if (m_sock == 0)
    {
      Print("[DQN] Send: socket is 0 (not initialized)");
      return false;
    }

    int slen = StringLen(text);
    if (slen <= 0)
    {
      Print("[DQN] Send: empty text");
      return false;
    }

    uchar payload[];

    // Let MQL allocate the buffer as needed
    int len = StringToCharArray(text, payload, 0, WHOLE_ARRAY);
    // StringToCharArray returns number of chars INCLUDING terminating 0

    if (len <= 0)
    {
      Print("[DQN] Send: StringToCharArray failed, len=", len);
      return false;
    }

    // Strip trailing null terminator if present
    if (payload[len - 1] == 0)
      len--;

    // Ensure array size matches the payload length we're going to send
    ArrayResize(payload, len);

    int sent = zmq_send(m_sock, payload, len, 0); // flags = 0 is correct
    if (sent != len)
    {
      // errno is not super reliable here, but we log anyway
      int err = zmq_errno();
      Print("[DQN] Send: zmq_send failed. sent=", sent, " len=", len, " errno=", err);
      return false;
    }

    return true;
  }

  bool Recv(string &out)
  {
    if (m_sock == 0)
      return false;

    uchar buf[];
    ArrayResize(buf, m_recv_cap);

    int n = zmq_recv(m_sock, buf, m_recv_cap, 0);
    if (n > 0)
    {
      out = CharArrayToString(buf, 0, n, CP_UTF8);
      return true;
    }
    return false;
  }

  void Shutdown()
  {
    if (m_sock != 0)
    {
      zmq_close(m_sock);
      m_sock = 0;
    }
    if (m_ctx != 0)
    {
      zmq_ctx_term(m_ctx);
      m_ctx = 0;
    }
  }
};

// ==================== Globals ====================

CZmqClient Zmq;
int LastBars = -1;
uint LastMs = 0;
int gLastResetDay = -1;    // last day-of-month we sent a reset
datetime gLastBarTime = 0; // last bar time we sent a decision for
datetime gLastLoggedBarTime = 0; // last bar time we logged any action
datetime gLastExecBarTime = 0; // last bar time we logged exec diagnostics

// ---- Exec diagnostics (per action) ----
int gLastActionRequested = -1;
int gLastActionEffective = -1;
int gLastPosBefore = 0;
double gLastPosAgeBars = 0.0;
int gLastMinHoldBars = 0;
int gLastCooldownBefore = 0;
bool gLastBlockedByCooldown = false;
bool gLastBlockedByMinHold = false;
bool gLastBlockedByReverse = false;

int gLastOrderSendAttempted = 0;
bool gLastOrderSendOk = false;
int gLastOrderSendTicket = -1;
int gLastOrderSendError = 0;

int gLastOrderCloseAttempted = 0;
bool gLastOrderCloseOk = false;
int gLastOrderCloseError = 0;

int gLastErrorAfter = 0;

// ---- Close diagnostics (from order history) ----
bool gLastCloseDetected = false;
string gLastCloseReason = "";
int gLastCloseTicket = -1;
double gLastClosePrice = 0.0;
datetime gLastCloseTime = 0;
double gLastCloseProfit = 0.0;
double gLastCloseSwap = 0.0;
double gLastCloseCommission = 0.0;

// ==================== Utilities ====================
int DigitsPips()
{
  int d = Digits; // price digits of current symbol
  if (d == 3 || d == 5)
    return 10; // 0.001 / 0.00001 symbols
  return 1;    // 0.01 / 0.0001 symbols
}

bool IsNewBar()
{
  datetime t = iTime(Symbol(), Period(), 0);
  if (t != g_last_bar_time)
  {
    g_last_bar_time = t;
    return (true);
  }
  return (false);
}

void OpenActionCsv()
{
  if (!LogActions)
    return;

  g_fh_actions = FileOpen(
      ActionsFile,
      FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ,
      ';');

  if (g_fh_actions == INVALID_HANDLE)
  {
    Print("ERROR opening CSV: ", GetLastError());
    return;
  }

  if (FileSize(g_fh_actions) == 0)
  {
    FileWrite(
        g_fh_actions,
        "bar_index",
        "bar_time",
        "action",
        "position_side",
        "ticket",
        "lots",
        "bid",
        "ask");
  }
  else
  {
    FileSeek(g_fh_actions, 0, SEEK_END);
  }
}

void ResetExecDiag()
{
  gLastActionRequested = -1;
  gLastActionEffective = -1;
  gLastPosBefore = 0;
  gLastPosAgeBars = 0.0;
  gLastMinHoldBars = 0;
  gLastCooldownBefore = 0;
  gLastBlockedByCooldown = false;
  gLastBlockedByMinHold = false;
  gLastBlockedByReverse = false;

  gLastOrderSendAttempted = 0;
  gLastOrderSendOk = false;
  gLastOrderSendTicket = -1;
  gLastOrderSendError = 0;

  gLastOrderCloseAttempted = 0;
  gLastOrderCloseOk = false;
  gLastOrderCloseError = 0;

  gLastErrorAfter = 0;

  gLastCloseDetected = false;
  gLastCloseReason = "";
  gLastCloseTicket = -1;
  gLastClosePrice = 0.0;
  gLastCloseTime = 0;
  gLastCloseProfit = 0.0;
  gLastCloseSwap = 0.0;
  gLastCloseCommission = 0.0;
}

void OpenExecCsv()
{
  if (!LogExec)
    return;

  g_fh_exec = FileOpen(
      ExecFile,
      FILE_CSV | FILE_WRITE | FILE_READ | FILE_SHARE_READ,
      ';');

  if (g_fh_exec == INVALID_HANDLE)
  {
    Print("ERROR opening Exec CSV: ", GetLastError());
    return;
  }

  if (FileSize(g_fh_exec) == 0)
  {
    FileWrite(
        g_fh_exec,
        "bar_index",
        "bar_time",
        "action_req",
        "action_eff",
        "pos_side_before",
        "pos_side_after",
        "pos_age_bars",
        "min_hold_bars",
        "cooldown_before",
        "blocked_by_cooldown",
        "blocked_by_min_hold",
        "blocked_by_reverse",
        "order_send_attempted",
        "order_send_ok",
        "order_send_ticket",
        "order_send_error",
        "order_close_attempted",
        "order_close_ok",
        "order_close_error",
        "last_error",
        "close_detected",
        "close_reason",
        "close_ticket",
        "close_price",
        "close_time",
        "close_profit",
        "close_swap",
        "close_commission");
  }
  else
  {
    FileSeek(g_fh_exec, 0, SEEK_END);
  }
}

void LogAction(
    int action,
    int pos_side,
    int ticket,
    double lots)
{
  if (!LogActions || g_fh_actions == INVALID_HANDLE)
    return;

  int bar_index = CurrentBarIndex();

  FileWrite(
      g_fh_actions,
      bar_index,
      TimeToString(iTime(Symbol(), Period(), 0), TIME_DATE | TIME_MINUTES),
      action,
      pos_side,
      ticket,
      DoubleToString(lots, 2),
      DoubleToString(Bid, Digits),
      DoubleToString(Ask, Digits));

  FileFlush(g_fh_actions);
}

void LogExecDiagnostics()
{
  if (!LogExec || g_fh_exec == INVALID_HANDLE)
    return;

  datetime bt = iTime(Symbol(), Period(), 0);
  if (gLastExecBarTime == bt)
    return;

  int bar_index = CurrentBarIndex();
  int pos_after = CurrentDirection();

  FileWrite(
      g_fh_exec,
      bar_index,
      TimeToString(bt, TIME_DATE | TIME_MINUTES),
      gLastActionRequested,
      gLastActionEffective,
      gLastPosBefore,
      pos_after,
      DoubleToString(gLastPosAgeBars, 2),
      gLastMinHoldBars,
      gLastCooldownBefore,
      (gLastBlockedByCooldown ? 1 : 0),
      (gLastBlockedByMinHold ? 1 : 0),
      (gLastBlockedByReverse ? 1 : 0),
      gLastOrderSendAttempted,
      (gLastOrderSendOk ? 1 : 0),
      gLastOrderSendTicket,
      gLastOrderSendError,
      gLastOrderCloseAttempted,
      (gLastOrderCloseOk ? 1 : 0),
      gLastOrderCloseError,
      gLastErrorAfter,
      (gLastCloseDetected ? 1 : 0),
      gLastCloseReason,
      gLastCloseTicket,
      DoubleToString(gLastClosePrice, Digits),
      TimeToString(gLastCloseTime, TIME_DATE | TIME_MINUTES),
      DoubleToString(gLastCloseProfit, 2),
      DoubleToString(gLastCloseSwap, 2),
      DoubleToString(gLastCloseCommission, 2));

  FileFlush(g_fh_exec);
  gLastExecBarTime = bt;
}

void OnOpenedPositionWithTicket(int ticket)
{
  if (ticket < 0)
    return;
  if (!OrderSelect(ticket, SELECT_BY_TICKET))
    return;

  int type = OrderType();
  double fill = OrderOpenPrice();

  // Capture spread at the entry moment using current tick prices as approximation.
  double ask0 = MarketInfo(Symbol(), MODE_ASK);
  double bid0 = MarketInfo(Symbol(), MODE_BID);
  double spread = ask0 - bid0;

  if (type == OP_BUY)
  {
    // Env stores long entry_ask (filled) and entry_bid (bid0).
    gEntryAsk = fill;
    gEntryBid = fill - spread;
  }
  else if (type == OP_SELL)
  {
    // Env stores short entry_bid (filled) and entry_ask (ask0).
    gEntryBid = fill;
    gEntryAsk = fill + spread;
  }
  gHasEntry = true;
}

void OnClosedPosition()
{
  if (gPrevDir != 0) // only if we truly were in a position
  {
    gCooldownBars = GetCooldownBars();
  }
  // Capture last closed order info for diagnostics
  gLastCloseDetected = true;
  gLastCloseReason = "unknown";
  gLastCloseTicket = -1;
  gLastClosePrice = 0.0;
  gLastCloseTime = 0;
  gLastCloseProfit = 0.0;
  gLastCloseSwap = 0.0;
  gLastCloseCommission = 0.0;

  int total = OrdersHistoryTotal();
  for (int i = total - 1; i >= 0; i--)
  {
    if (!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
      continue;
    if (OrderSymbol() != Symbol() || OrderMagicNumber() != Magic)
      continue;
    int type = OrderType();
    if (type != OP_BUY && type != OP_SELL)
      continue;

    datetime ct = OrderCloseTime();
    if (ct <= 0)
      continue;
    if (ct >= gLastCloseTime)
    {
      gLastCloseTime = ct;
      gLastCloseTicket = OrderTicket();
      gLastClosePrice = OrderClosePrice();
      gLastCloseProfit = OrderProfit();
      gLastCloseSwap = OrderSwap();
      gLastCloseCommission = OrderCommission();

      double tp = OrderTakeProfit();
      double sl = OrderStopLoss();
      double tol = MarketInfo(Symbol(), MODE_POINT) * 2;
      if (tp > 0 && MathAbs(gLastClosePrice - tp) <= tol)
        gLastCloseReason = "tp";
      else if (sl > 0 && MathAbs(gLastClosePrice - sl) <= tol)
        gLastCloseReason = "sl";
      else
        gLastCloseReason = "manual";
    }
  }
  gHasEntry = false;
  gEntryAsk = 0.0;
  gEntryBid = 0.0;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double PipToPrice(int pips)
{
  double point = MarketInfo(Symbol(), MODE_POINT);
  return pips * DigitsPips() * point;
}

void PushTickHistory(double ask, double bid)
{
  if (!SendTickHistory)
    return;

  int n = ArraySize(gTickAsk);
  if (n >= MaxTickHistory && MaxTickHistory > 0)
  {
    // Drop oldest element to cap size
    for (int i = 1; i < n; i++)
    {
      gTickAsk[i - 1] = gTickAsk[i];
      gTickBid[i - 1] = gTickBid[i];
    }
    n = n - 1;
    ArrayResize(gTickAsk, n);
    ArrayResize(gTickBid, n);
  }

  ArrayResize(gTickAsk, n + 1);
  ArrayResize(gTickBid, n + 1);
  gTickAsk[n] = ask;
  gTickBid[n] = bid;
}

string JsonFromArrayDouble(const double &arr[])
{
  int n = ArraySize(arr);
  string s = "[";
  for (int i = 0; i < n; i++)
  {
    s += DoubleToString(arr[i], 8);
    if (i < n - 1)
      s += ",";
  }
  s += "]";
  return s;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string JsonBuildStep(const double &bw[][MAX_FEATURES], double ask, double bid)
{
  int L = ArrayRange(bw, 0);
  int N = 5; // O,H,L,C,V

  string jw = "[";
  for (int i = 0; i < L; i++)
  {
    jw += "[";
    for (int j = 0; j < N; j++)
    {
      jw += DoubleToString(bw[i][j], 8);
      if (j < N - 1)
        jw += ",";
    }
    jw += "]";
    if (i < L - 1)
      jw += ",";
  }
  jw += "]";

  int pos_side = CurrentDirection();
  double pos_age_bars = CurrentAgeBars();
  double lot = CurrentLot();
  if (ParityMode)
  {
    lot = ParityLot;
  }
  else
  {
    if (lot <= 0.0)
      lot = CalculatePositionSize(RiskPercent, SlPips);
  }
  double pip_dec = PipDecimalValue();
  double ex_rate = 1.0;
  int bar_index = CurrentBarIndex();
  string bar_time = TimeToString(iTime(Symbol(), Period(), 0), TIME_DATE | TIME_MINUTES);

  // ---- Entry prices (spread-aware) ----
  // Prefer stored entry_ask/entry_bid; fallback if missing.
  string entryAskJs = "null";
  string entryBidJs = "null";
  string entryPriceJs = "null";

  if (pos_side != 0)
  {
    double op = CurrentEntryPrice(); // OrderOpenPrice
    entryPriceJs = DoubleToString(op, Digits);

    if (gHasEntry)
    {
      entryAskJs = DoubleToString(gEntryAsk, Digits);
      entryBidJs = DoubleToString(gEntryBid, Digits);
    }
    else
    {
      // Fallback: approximate using current spread (not perfect, but avoids None)
      double spr = ask - bid;
      if (pos_side > 0) // long: open price ~ ask
      {
        entryAskJs = DoubleToString(op, Digits);
        entryBidJs = DoubleToString(op - spr, Digits);
      }
      else // short: open price ~ bid
      {
        entryBidJs = DoubleToString(op, Digits);
        entryAskJs = DoubleToString(op + spr, Digits);
      }
    }
  }

  string js = "{\"cmd\":\"step\""
              ",\"bar_window_raw\":" +
              jw +
              ",\"ask\":" + DoubleToString(ask, Digits) +
              ",\"bid\":" + DoubleToString(bid, Digits) +
              ",\"bar_index\":" + IntegerToString(bar_index) +
              ",\"bar_time\":\"" + bar_time + "\"" +
              ",\"position_side\":" + IntegerToString(pos_side) +
              ",\"entry_price\":" + entryPriceJs + // keep legacy
              ",\"entry_ask\":" + entryAskJs +     // NEW
              ",\"entry_bid\":" + entryBidJs +     // NEW
              ",\"pos_age_bars\":" + DoubleToString(pos_age_bars, 2) +
              ",\"pip_decimal\":" + DoubleToString(pip_dec, 8) +
              ",\"sl_pips\":" + IntegerToString(SlPips) +
              ",\"lot\":" + DoubleToString(lot, 2) +
              ",\"exchange_rate\":" + DoubleToString(ex_rate, 4) +
              "}";

  if (SendTickHistory)
  {
    int n = ArraySize(gTickAsk);
    if (n > 0 && n == ArraySize(gTickBid))
    {
      js = StringSubstr(js, 0, StringLen(js) - 1) +
           ",\"tick_asks\":" + JsonFromArrayDouble(gTickAsk) +
           ",\"tick_bids\":" + JsonFromArrayDouble(gTickBid) +
           "}";
    }
  }
  return js;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int ParseActionA(const string reply)
{
  Print("[DQN] raw reply = ", reply); // already had this before

  int pos = StringFind(reply, "\"a\":");
  if (pos < 0)
  {
    Print("[DQN] 'a' not found, holding");
    return -1;
  }

  string tail = StringSubstr(reply, pos + 4);
  int comma = StringFind(tail, ",");
  string num = (comma > 0 ? StringSubstr(tail, 0, comma) : tail);
  int a = (int)StringToInteger(num);

  if (a < 0 || a > 2)
  {
    Print("[DQN] invalid a=", a);
    return -1;
  }

  string name = (a == 0 ? "HOLD" : (a == 1 ? "BUY" : "SELL"));
  Print("[DQN] parsed action a=", a, " (", name, ")");
  return a;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool BuildBarFeaturesWindow(const int L, double &out[][MAX_FEATURES])
{
  ArrayResize(out, L);

  // We want OLDEST -> NEWEST ordering in the JSON
  // We'll fill k=0 oldest, k=L-1 newest (shift 0)
  for (int k = 0; k < L; k++)
  {
    // Use previous closed bar window (shift +1) to avoid current-bar lookahead.
    int shift = (L - 1) - k + 1;

    double o = iOpen(Symbol(), Period(), shift);
    double h = iHigh(Symbol(), Period(), shift);
    double l = iLow(Symbol(), Period(), shift);
    double c = iClose(Symbol(), Period(), shift);
    double v = iVolume(Symbol(), Period(), shift);

    out[k][0] = o;
    out[k][1] = h;
    out[k][2] = l;
    out[k][3] = c;
    out[k][4] = v;
  }
  return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CountPositions(int dir) // 0 BUY, 1 SELL (this EA's magic only)
{
  int cnt = 0;
  for (int i = 0; i < OrdersTotal(); i++)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
      {
        if ((dir == 0 && OrderType() == OP_BUY) || (dir == 1 && OrderType() == OP_SELL))
          cnt++;
      }
    }
  }
  return cnt;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ExecuteAction(const int a)
{
  // Action semantics aligned with TradingEnv / training:
  //   If FLAT:      0=WAIT, 1=OPEN_LONG, 2=OPEN_SHORT
  //   If IN TRADE:  0=HOLD, 1=CLOSE,     2=REVERSE (close + open opposite)
  //
  // NOTE: allow_reverse is controlled on the Python side; if disabled, the model
  // should not output 2 while in position. We still implement REVERSE here.

  int dir = CurrentDirection(); // -1 short, 0 flat, 1 long
  ResetLastError();
  gLastActionRequested = a;
  gLastActionEffective = a;
  gLastPosBefore = dir;
  gLastPosAgeBars = (dir == 0 ? 0.0 : CurrentAgeBars());
  gLastMinHoldBars = GetMinHoldBars();
  gLastCooldownBefore = gCooldownBars;

  if (a == 0) // WAIT/HOLD
  {
    gLastActionEffective = 0;
    return true;
  }

  double ask = MarketInfo(Symbol(), MODE_ASK);
  double bid = MarketInfo(Symbol(), MODE_BID);
  double sl = PipToPrice(SlPips);
  double tp = PipToPrice(TpPips);

  // -------- FLAT actions --------
  if (dir == 0)
  {
    if (gCooldownBars > 0)
    {
      // during cooldown, ignore BUY/SELL entries
      if (a == 1 || a == 2)
      {
        Print("[DQN] COOLDOWN: action=", a, " -> HOLD (cooldown_bars=", gCooldownBars, ")");
        gLastBlockedByCooldown = true;
        gLastActionEffective = 0;
        return true;
      }
    }
    if (a == 1)
    {
      double lot = (ParityMode ? ParityLot : CalculatePositionSize(RiskPercent, SlPips));
      gLastOrderSendAttempted = 1;
      ResetLastError();
      int ticket = OrderSend(Symbol(), OP_BUY, lot, ask, SlippagePoints,
                             ask - sl, ask + tp, "DQN BUY", Magic, 0, clrGreen);
      gLastOrderSendError = GetLastError();
      gLastErrorAfter = gLastOrderSendError;
      // entry capture is done after successful OrderSend
      if (ticket < 0)
      {
        Print("[DQN] BUY failed, error=", GetLastError());
        gLastActionEffective = 0;
        return false;
      }
      gLastOrderSendOk = true;
      gLastOrderSendTicket = ticket;
      OnOpenedPositionWithTicket(ticket);
      return true;
    }
    if (a == 2)
    {
      double lot = (ParityMode ? ParityLot : CalculatePositionSize(RiskPercent, SlPips));
      gLastOrderSendAttempted = 1;
      ResetLastError();
      int ticket = OrderSend(Symbol(), OP_SELL, lot, bid, SlippagePoints,
                             bid + sl, bid - tp, "DQN SELL", Magic, 0, clrTomato);
      gLastOrderSendError = GetLastError();
      gLastErrorAfter = gLastOrderSendError;
      // entry capture is done after successful OrderSend
      if (ticket < 0)
      {
        Print("[DQN] SELL failed, error=", GetLastError());
        gLastActionEffective = 0;
        return false;
      }
      gLastOrderSendOk = true;
      gLastOrderSendTicket = ticket;
      OnOpenedPositionWithTicket(ticket);
      return true;
    }
    return true;
  }

  // -------- IN POSITION actions --------
  // Enforce min-hold and disable reverse if configured to mirror training env.
  double age_bars = CurrentAgeBars();

  // If we haven't held long enough, force HOLD for CLOSE/REVERSE attempts
  if (age_bars < GetMinHoldBars())
  {
    if (a == 1 || a == 2)
    {
      Print("[DQN] GATED: action=", a, " -> HOLD (age_bars=", DoubleToString(age_bars, 2),
            " < GetMinHoldBars()=", GetMinHoldBars(), ")");
      gLastBlockedByMinHold = true;
      gLastActionEffective = 0;
      return true; // HOLD
    }
  }

  if (a == 1)
  {
    // CLOSE (only allowed once min-hold satisfied)
    if (!CloseCurrentPosition())
    {
      gLastActionEffective = 0;
      return false;
    }
    return true;
  }

  if (a == 2)
  {
    // REVERSE disabled: treat as HOLD
    if (!GetAllowReverse())
    {
      Print("[DQN] Reverse disabled -> HOLD");
      gLastBlockedByReverse = true;
      gLastActionEffective = 0;
      return true;
    }

    // REVERSE: close, then open opposite
    if (!CloseCurrentPosition())
    {
      gLastActionEffective = 0;
      return false;
    }

    // After close, open opposite of previous dir
    if (dir == 1)
    {
      double lot = (ParityMode ? ParityLot : CalculatePositionSize(RiskPercent, SlPips));
      int ticket = OrderSend(Symbol(), OP_SELL, lot, bid, SlippagePoints,
                             bid + sl, bid - tp, "DQN REVERSE SELL", Magic, 0, clrTomato);
      if (ticket < 0)
      {
        Print("[DQN] REVERSE SELL failed, error=", GetLastError());
        return false;
      }
      return true;
    }
    if (dir == -1)
    {
      double lot = (ParityMode ? ParityLot : CalculatePositionSize(RiskPercent, SlPips));
      int ticket = OrderSend(Symbol(), OP_BUY, lot, ask, SlippagePoints,
                             ask - sl, ask + tp, "DQN REVERSE BUY", Magic, 0, clrGreen);
      if (ticket < 0)
      {
        Print("[DQN] REVERSE BUY failed, error=", GetLastError());
        return false;
      }
      return true;
    }
  }

  return true;
}

int CurrentDirection()
{
  int dir = 0;
  for (int i = 0; i < OrdersTotal(); i++)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
      {
        if (OrderType() == OP_BUY)
          return 1;
        if (OrderType() == OP_SELL)
          return -1;
      }
    }
  }
  return dir;
}

int CurrentBarIndex()
{
  int shift = iBarShift(Symbol(), Period(), iTime(Symbol(), Period(), 0), true);
  if (shift < 0)
    return -1;
  int total = iBars(Symbol(), Period());
  return (total - 1 - shift);
}

int CurrentTicket()
{
  for (int i = 0; i < OrdersTotal(); i++)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
        return OrderTicket();
    }
  }
  return -1;
}

datetime CurrentEntryTime()
{
  for (int i = 0; i < OrdersTotal(); i++)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
        return OrderOpenTime();
    }
  }
  return 0;
}

void SyncPositionTransitions()
{
  int curDir = CurrentDirection();
  int curTicket = CurrentTicket();
  datetime curEntry = CurrentEntryTime();

  // If we were in a position and now flat => close happened (TP/SL/manual/etc.)
  if (gPrevDir != 0 && curDir == 0)
  {
    OnClosedPosition();
  }

  // If we were flat and now in a position => open happened
  if (gPrevDir == 0 && curDir != 0)
  {
    OnOpenedPositionWithTicket(curTicket);
  }

  gPrevDir = curDir;
  gPrevTicket = curTicket;
  gPrevEntryTime = curEntry;
}

void RefreshPrevPositionState()
{
  gPrevDir = CurrentDirection();
  gPrevTicket = CurrentTicket();
  gPrevEntryTime = CurrentEntryTime();
}

//+------------------------------------------------------------------+
//| Current position info for observation alignment (EA -> server)   |
//+------------------------------------------------------------------+
double CurrentEntryPrice()
{
  for (int i = 0; i < OrdersTotal(); i++)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
      {
        return OrderOpenPrice();
      }
    }
  }
  return -1.0;
}

double CurrentLot()
{
  for (int i = 0; i < OrdersTotal(); i++)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
      {
        return OrderLots();
      }
    }
  }
  return 0.0;
}

double CurrentAgeBars()
{
  // Approximate age in bars since entry (0 = opened in current bar)
  for (int i = 0; i < OrdersTotal(); i++)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
      {
        datetime ot = OrderOpenTime();
        int shift = iBarShift(Symbol(), Period(), ot, true);
        if (shift < 0)
          shift = 0;
        return (double)shift;
      }
    }
  }
  return 0.0;
}

double PipDecimalValue()
{
  double point = MarketInfo(Symbol(), MODE_POINT);
  return DigitsPips() * point;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// Close any open position for this EA & symbol
bool CloseCurrentPosition()
{
  gLastOrderCloseAttempted = 0;
  gLastOrderCloseOk = false;
  gLastOrderCloseError = 0;

  for (int i = OrdersTotal() - 1; i >= 0; i--)
  {
    if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
    {
      if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
      {
        int type = OrderType();
        double lots = OrderLots();
        int ticket = OrderTicket();
        double price = (type == OP_BUY)
                           ? MarketInfo(Symbol(), MODE_BID)
                           : MarketInfo(Symbol(), MODE_ASK);

        gLastOrderCloseAttempted = 1;
        ResetLastError();
        if (!OrderClose(ticket, lots, price, SlippagePoints, clrRed))
        {
          gLastOrderCloseError = GetLastError();
          gLastErrorAfter = gLastOrderCloseError;
          Print("[DQN] Failed to close position, ticket=", ticket,
                " error=", GetLastError());
          return false;
        }
        gLastOrderCloseError = GetLastError();
        gLastErrorAfter = gLastOrderCloseError;
        gLastOrderCloseOk = true;
        OnClosedPosition();
        return true; // we only expect one position
      }
    }
  }
  return true; // nothing to close
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CalculatePositionSize(double risk_percent, int sl_pips)
{
  double balance = AccountBalance();
  double risk_usd = balance * (risk_percent / 100.0);

  double pip_value_per_lot = MarketInfo(Symbol(), MODE_TICKVALUE) * 10;
  // For most FX pairs: 1 pip = 10 ticks

  if (pip_value_per_lot <= 0)
    pip_value_per_lot = 10.0; // fallback for safety

  double value_per_lot_sl = sl_pips * pip_value_per_lot;

  double lot = risk_usd / value_per_lot_sl;

  // Normalize to broker limits:
  double minlot = MarketInfo(Symbol(), MODE_MINLOT);
  double lotstep = MarketInfo(Symbol(), MODE_LOTSTEP);
  double maxlot = MarketInfo(Symbol(), MODE_MAXLOT);

  lot = MathMax(minlot, MathMin(lot, maxlot));
  lot = NormalizeDouble(lot / lotstep, 0) * lotstep;

  return lot;
}

// ==================== EA Lifecycle ====================
int OnInit()
{
  Print("[DQN] MT4 EA starting...");
  if (!Zmq.Init(Endpoint))
  {
    Print("[DQN] ZMQ connect failed: ", Endpoint);
    return (INIT_FAILED);
  }
  // send reset
  if (!Zmq.Send("{\"cmd\":\"reset\"}"))
    Print("[DQN] reset send failed");
  string rep;
  Zmq.Recv(rep);
  // ArrayResize(BarWin, WindowLen);
  // for (int i = 0; i < WindowLen; i++)
  //   ArrayResize(BarWin, NFeatures);
  ArrayResize(BarWin, RawWindowLen);
  LastBars = iBars(Symbol(), Period());
  LastMs = GetTickCount();
  gLastResetDay = TimeDay(TimeCurrent()); // remember reset day

  g_last_bar_time = 0;
  gLastLoggedBarTime = 0;
  OpenActionCsv();
  OpenExecCsv();
  gLastExecBarTime = 0;

  // Initialize transition tracker
  gPrevDir = CurrentDirection();
  gPrevTicket = CurrentTicket();
  gPrevEntryTime = CurrentEntryTime();

  gCooldownBars = 0; // CRITICAL: ensure no phantom cooldown at startup
  ArrayResize(gTickAsk, 0);
  ArrayResize(gTickBid, 0);

  return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
  if (g_fh_actions != INVALID_HANDLE)
    FileClose(g_fh_actions);
  if (g_fh_exec != INVALID_HANDLE)
    FileClose(g_fh_exec);

  // Tell server to shut down (if connected)
  if (Zmq.Send("{\"cmd\":\"shutdown\"}"))
  {
    string rep;
    Zmq.Recv(rep); // ignore result, just best-effort
  }
  Zmq.Shutdown();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
  int bars_now = iBars(Symbol(), Period());
  if (bars_now < RawWindowLen + 2)
  {
    Print("[DQN] Not enough bars: bars_now=", bars_now, " need>=", (RawWindowLen + 2),
          " RawWindowLen=", RawWindowLen);
    return;
  }

  double ask_tick = MarketInfo(Symbol(), MODE_ASK);
  double bid_tick = MarketInfo(Symbol(), MODE_BID);
  PushTickHistory(ask_tick, bid_tick);

  // HARD BAR GATE - guarantees exactly 1 decision per bar
  if (!IsNewBar())
    return;

  // Current wall-clock for this tick
  uint now = GetTickCount();

  // --- 0) Throttle only in live/forward trading, NOT in backtest ---
  if (!MQLInfoInteger(MQL_TESTER))
  {
    static uint lastTickMs = 0;
    if (now - lastTickMs < (uint)StepMs)
      return;
    lastTickMs = now;
  }

  // Sync transitions so cooldown also applies to TP/SL closes (env parity)
  ResetExecDiag();
  SyncPositionTransitions();

  // NOTE: cooldown decrement happens AFTER action processing (to match TradingEnv ordering)

  // --- 0.5) Daily reset: mimic new episode each trading day ---
  // --- 0.5) Daily reset: optional; disable in ParityMode to match offline eval unless you also reset there ---
  int today = TimeDay(TimeCurrent());
  if (!(ParityMode && ParityDisableDailyReset) && gLastResetDay != today)
  {
    if (!Zmq.Send("{\"cmd\":\"reset\"}"))
      Print("[DQN] daily reset send failed");
    else
    {
      string rep;
      Zmq.Recv(rep); // wait for "reset_done" from server
      Print("[DQN] daily reset ok, reply=", rep);
      gLastResetDay = today;
    }
  }

  // --- 1) Build features and SEND a new request (blocking receive) ---

  // Ensure we have enough history to fill a full window
  bars_now = iBars(Symbol(), Period());
  if (bars_now < RawWindowLen + 2)
    return;

  LastBars = bars_now;

  if (!BuildBarFeaturesWindow(RawWindowLen, BarWin))
  {
    Print("[DQN] raw window build failed");
    return;
  }

  double ask = UseTickAskBid ? MarketInfo(Symbol(), MODE_ASK)
                             : iClose(Symbol(), Period(), 0);
  double bid = UseTickAskBid ? MarketInfo(Symbol(), MODE_BID)
                             : iClose(Symbol(), Period(), 0);

  // Serialize OHLCV window as bar_window_raw and send to server
  string req = JsonBuildStep(BarWin, ask, bid);

  if (!Zmq.Send(req))
  {
    Print("[DQN] send failed");
    return;
  }

  gLastSendMs = now;

  string reply;
  if (!Zmq.Recv(reply))
  {
    Print("[DQN] recv failed");
    return;
  }

  int a = ParseActionA(reply);
  if (a < 0)
    return;

  bool ok = ExecuteAction(a);
  if (!ok)
    Print("[DQN] action exec failed: ", a);

  // ---- compute current position AFTER execution ----
  datetime bt = iTime(Symbol(), Period(), 0);
  if (gLastLoggedBarTime != bt)
  {
    int pos_side = CurrentDirection(); // -1,0,1
    int ticket = -1;
    double lots = 0.0;
    for (int i = OrdersTotal() - 1; i >= 0; i--)
    {
      if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
        if (OrderSymbol() == Symbol() && OrderMagicNumber() == Magic)
        {
          ticket = OrderTicket();
          lots = OrderLots();
          break;
        }
      }
    }
    LogAction(a, pos_side, ticket, lots);
    gLastLoggedBarTime = bt;
  }

  LogExecDiagnostics();
  RefreshPrevPositionState();

  // Cooldown decrement after decision to match TradingEnv ordering
  if (gCooldownBars > 0)
    gCooldownBars--;

  // Reset tick history after request so we only send new ticks next time
  ArrayResize(gTickAsk, 0);
  ArrayResize(gTickBid, 0);
}

//+------------------------------------------------------------------+
