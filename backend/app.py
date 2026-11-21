# -*- coding: utf-8 -*-
"""
future 后端服务：单文件 HTTP 服务，提供 JSON API + 托管前端静态页。

启动：python app.py
端口：8103

API 一览：
  GET /api/basis?variety=IC          基差看板（当前四合约，日线口径）
  GET /api/history?variety=IC&years=1  历史基差/年化序列 + 分位统计
  GET /api/move?variety=IC&years=1   移仓助手（合约对价差分析）
  GET /api/pair?near=IC2610&far=IC2611&years=1  单个合约对详情
  GET /api/realtime?variety=IC       单个品种实时行情（前端按需调用）
  GET /api/realtime_all              四个品种实时行情（首屏专用，1 次拿全）
  GET /api/meta                      品种清单 / 数据最新日期 / 是否允许一键更新
  POST /api/refresh                  后台抓取最新数据（仅本机开放，见 ALLOW_REFRESH）
  GET /api/refresh_status            查询抓取进度与日志
"""
import os
import re
import sys
import json
import time
import threading
import subprocess
import traceback
from datetime import datetime
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(ROOT, "core")
WEB_DIR = os.path.join(ROOT, "frontend")
sys.path.insert(0, CORE)

from config import VARIETIES, ENABLED_VARIETIES, DEFAULT_PORT
import engine
import data_source as ds
import realtime as rt
import ironore


def json_resp(handler, obj, status=200):
    body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    # API 一律不缓存：数据是实时算的，缓存会让前端拿到过期的 allow_refresh / 行情
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _today_str():
    """返回今天的 YYYY-MM-DD。实时口径一律按「今天」判合约是否还在交易。"""
    return datetime.now().date().strftime("%Y-%m-%d")


def _realtime_symbols(variety):
    """实时口径的合约清单：按今天筛选，已摘牌合约不进清单。

    ⚠️ 口径必须与历史接口分开，别混用：
      · /api/basis、/api/history、/api/pair 是历史回溯，as_of 必须取「数据最新日」，
        否则数据停更时历史序列会错位；
      · /api/realtime* 和 /api/move 的「当前」是给今天看盘/下单用的，必须按今天。

    data_source.get_active_symbols 的 as_of 默认取数据里最新日期。一旦数据停更
    （例：本机 6 周没更新，latest=2026-09-30），最后交易日 10-16 的 IC2610
    仍会被判「在交易」；而 days_left 又按今天算 → -28 天、年化空、基差拿摘牌
    冻结价减今日现货，页面上就出现一整行无意义数据（2026-11-13 实测）。
    """
    return ds.get_active_symbols(variety, as_of=_today_str())


def build_realtime_payload(variety, quotes=None, spot_q=None):
    """
    组装单个品种的实时行情 payload。

    quotes/spot_q 可由调用方传入（批量接口会一次性拉完四个品种再分发），
    不传则自己抓（单品种接口的用法）。

    抽成独立函数的原因：/api/realtime 和 /api/realtime_all 要共用同一套
    基差/年化计算逻辑，绝不能写两份 —— 写两份迟早会漂移，两处数字对不上。
    """
    cfg = VARIETIES.get(variety, {})
    syms = _realtime_symbols(variety)
    if quotes is None:
        quotes = rt.fetch_futures_quote(syms)
    if spot_q is None:
        spot_q = rt.fetch_index_quote(cfg.get("spot_code"))
    spot_price = spot_q["price"] if spot_q else None

    # 用实时价重算基差与年化
    qmap = {q["symbol"]: q for q in quotes}
    rows = []
    for s in syms:
        q = qmap.get(s)
        if not q or q["price"] is None:
            continue
        ltd = ds.symbol_last_trade_date(s)
        days_left = None
        if ltd:
            # 注意：必须用纯日期相减，不能用 datetime.now()（含时分秒会导致少算1天）
            _today = datetime.now().date()
            _ltd = datetime.strptime(ltd, "%Y-%m-%d").date()
            days_left = (_ltd - _today).days
        # 防御闸门：实时看板不展示已到期合约。即使上游口径再漂移、或行情源
        # 返回摘牌合约的冻结价，也不会渲染出负基差 / 空年化的脏行。
        if days_left is not None and days_left <= 0:
            continue
        basis = None
        basis_pct = None
        annual = None
        if spot_price:
            basis = spot_price - q["price"]
            basis_pct = basis / spot_price * 100
            if days_left and days_left > 0:
                annual = basis_pct * 365 / days_left
        rows.append({
            "symbol": s,
            "price": q["price"],
            "change": q["change"],
            "change_pct": q["change_pct"],
            "pre_close": q["pre_close"],
            "hold": q["hold"],
            "basis": round(basis, 2) if basis is not None else None,
            "basis_pct": round(basis_pct, 2) if basis_pct is not None else None,
            "annual": round(annual, 2) if annual is not None else None,
            "days_left": days_left,
        })
    # 已实现波动率：给「折合年化」配一个风险分母。
    # 年化贴水单独看没有意义 —— 23% 的贴水在 20% 波动的市场里是厚利，
    # 在 40% 波动的市场里只是风险补偿。carry/波动 = 年化 ÷ 波动。
    vol = engine.realized_vol(variety)
    if vol is not None:
        for r in rows:
            r["carry_vol"] = engine.carry_per_vol(r.get("annual"), vol)
    return {
        "ok": True, "variety": variety,
        "name": cfg.get("name", ""),
        "spot": spot_q,
        "rows": rows,
        "spot_percentile": engine.spot_percentile(variety, years=3, current_value=spot_price),
        "vol": round(vol, 2) if vol is not None else None,
        "vol_window": 20,
        "is_trading": rt.is_trading_time(),
        "is_live": bool(spot_price is not None and len(rows) > 0),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _live_fixed_tenor(variety, tenor_days):
    """
    用后台聚合的实时快照，把「今天的合约曲线」插值到固定期限。

    不额外发外网请求：/api/realtime_all 的后台线程每 10 秒已经把四个品种
    全部在交易合约的 basis_pct / days_left 放进 _RT_CACHE，这里只读内存。
    拿不到快照（冷启动首轮还没跑完）就返回 None，前端退回用日线末点。
    """
    with _RT_LOCK:
        payload = _RT_CACHE.get("payload")
    if not payload:
        return None
    vd = (payload.get("varieties") or {}).get(variety)
    if not vd:
        return None
    pts = []
    for r in vd.get("rows") or []:
        dl, bp = r.get("days_left"), r.get("basis_pct")
        if dl is None or bp is None or dl <= 0:
            continue
        pts.append({"symbol": r["symbol"], "days_left": dl, "basis_pct": bp})
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p["days_left"])
    ip = engine.interpolate_curve(pts, tenor_days)
    if ip is None:
        return None
    return ip["basis_pct"] * 365 / int(tenor_days)


# ============ 数据更新（页面一键抓取，仅本机开放） ============
# 为什么必须用环境变量控制，而不能靠"请求来自 127.0.0.1"判断：
#   线上是 Cloudflare → ARM nginx → 127.0.0.1:8103 反代，
#   后端看到的**所有**请求都来自 127.0.0.1，这个判据完全无效。
#   而抓取要枚举 4 个品种从上市到次年的全部月份合约（实测 600 个），
#   逐个发 HTTP 请求，在 1 核 ARM 上要跑几分钟 —— 对读者开放等于自杀。
#   所以线上 systemd 不设 ALLOW_REFRESH（默认 0），接口直接 403；
#   本地 start.bat 设 ALLOW_REFRESH=1。
ALLOW_REFRESH = os.environ.get("ALLOW_REFRESH", "0") == "1"

_REFRESH_LOCK = threading.Lock()
_REFRESH_LOG_MAX = 400
_refresh = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "iron_rc": None,
    "new_rows": None,
    "error": None,
    "stage": None,      # 当前在跑哪一步（前端提示用）
    "log": [],
}


def _refresh_snapshot():
    """当前抓取状态快照。log 只回传尾部若干行，避免响应过大。"""
    with _REFRESH_LOCK:
        snap = {
            "running": _refresh["running"],
            "started_at": _refresh["started_at"],
            "finished_at": _refresh["finished_at"],
            "returncode": _refresh["returncode"],
            "iron_rc": _refresh["iron_rc"],
            "new_rows": _refresh["new_rows"],
            "error": _refresh["error"],
            "stage": _refresh["stage"],
            "log": _refresh["log"][-60:],
        }
    snap["elapsed"] = (round(time.time() - snap["started_at"], 1)
                       if snap["running"] and snap["started_at"] else None)
    return snap


def _run_step(script, extra=None, stage=None):
    """跑一个子步骤，stdout 逐行收进刷新日志。返回退出码（异常返回 -1）。

    `script` 是相对 ROOT 的路径片段（如 ["backend", "fetch_data.py"]），
    `extra` 是命令行参数（如 ["--full"]）—— 两者必须分开，
    否则 os.path.join 会把参数当成路径拼上去（2026-10-04 实测踩到）。
    """
    if stage:
        with _REFRESH_LOCK:
            _refresh["stage"] = stage
    cmd = [sys.executable, os.path.join(ROOT, *script)] + list(extra or [])
    env = dict(os.environ)
    env["NO_PROXY"] = env.get("NO_PROXY", "127.0.0.1,localhost")
    env["PYTHONIOENCODING"] = "utf-8"      # 否则 Windows 下中文日志乱码
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            with _REFRESH_LOCK:
                _refresh["log"].append(line)
                if len(_refresh["log"]) > _REFRESH_LOG_MAX:
                    del _refresh["log"][0]
        return proc.wait()
    except Exception as e:
        traceback.print_exc()
        with _REFRESH_LOCK:
            _refresh["log"].append(f"步骤异常：{e}")
        return -1


def _refresh_worker(full=False):
    """后台线程：① 抓股指+铁矿期货日线 ② 重建铁矿石序列 ③ 清进程内缓存。

    必须异步 —— 抓取要几分钟，同步等会直接撞上前端 20 秒超时。
    """
    try:
        rc = _run_step(["backend", "fetch_data.py"], ["--full"] if full else [],
                       stage="股指与期货日线")

        new_rows = None
        with _REFRESH_LOCK:
            for line in _refresh["log"]:
                m = re.search(r"本次新写入期货日线\s*(\d+)\s*行", line)
                if m:
                    new_rows = int(m.group(1))

        # 第二步：重建铁矿石序列（重抓生意社基差率 → 反推现货 → 落盘 json）。
        # 依赖上一步刚入库的期货日线，所以必须放在后面。
        # 失败不回滚、不影响股指 —— 旧的 ironore_series.json 原样保留，页面照常能看，
        # 只是图还停在上一次的数据（前端会看到 error 提示）。
        iron_rc = None
        if rc == 0:
            with _REFRESH_LOCK:
                _refresh["log"].append("———— 重建铁矿石序列（生意社基差率 → 反推现货）————")
            iron_rc = _run_step(["scripts", "build_ironore_series.py"], ["--fetch"],
                                stage="铁矿石序列")

        # 关键一步：清掉进程内缓存，否则页面还是旧数据 —— 等于白跑
        ds.clear_caches()
        engine.clear_caches()
        ironore.clear_cache()

        with _REFRESH_LOCK:
            _refresh["returncode"] = rc
            _refresh["iron_rc"] = iron_rc
            _refresh["new_rows"] = new_rows
            if rc != 0:
                _refresh["error"] = f"抓取脚本退出码 {rc}"
            elif iron_rc != 0:
                _refresh["error"] = "股指数据已更新，但铁矿石重建失败（铁矿石图仍为上一次的数据）"
            else:
                _refresh["error"] = None
    except Exception as e:
        traceback.print_exc()
        with _REFRESH_LOCK:
            _refresh["error"] = str(e)
    finally:
        with _REFRESH_LOCK:
            _refresh["running"] = False
            _refresh["stage"] = None
            _refresh["finished_at"] = time.time()


class Handler(SimpleHTTPRequestHandler):
    # 关键：SimpleHTTPRequestHandler 默认是 HTTP/1.0 —— 每个请求都要新建 TCP 连接、
    # 用完即断。而浏览器（尤其 Edge）按 HTTP/1.1 的 keep-alive 复用连接，
    # 于是出现大量 ConnectionReset / ConnectionAborted，浏览器不断重试，
    # 页面表现为"卡在加载中"。改成 1.1 后连接可复用，加载速度恢复正常。
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        """给 HTML 关掉缓存。

        2026-10-04 真实踩到：新加的「更新数据」按钮死活刷新不出来 ——
        SimpleHTTPRequestHandler 不发 Cache-Control，浏览器就按启发式规则缓存，
        用户看到的还是旧页面。这里显式声明 no-cache，改完前端刷新即可生效。
        """
        if self.path.endswith(".html") or self.path.endswith(".htm"):
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # 静音

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        q = parse_qs(u.query)
        try:
            if path.startswith("/api/"):
                return self.handle_api(path, q)
            # 静态文件
            if path == "/":
                self.path = "/index.html"
            return super().do_GET()
        except Exception as e:
            traceback.print_exc()
            return json_resp(self, {"ok": False, "error": str(e)}, 500)

    def do_POST(self):
        """只有数据更新是 POST；其余一律 404。"""
        u = urlparse(self.path)
        try:
            if u.path == "/api/refresh":
                return self.handle_refresh(parse_qs(u.query))
            return json_resp(self, {"ok": False, "error": "unknown api"}, 404)
        except Exception as e:
            traceback.print_exc()
            return json_resp(self, {"ok": False, "error": str(e)}, 500)

    def handle_refresh(self, q):
        """启动一次后台抓取。已在跑就直接返回当前状态，不重复起线程。"""
        if not ALLOW_REFRESH:
            return json_resp(self, {
                "ok": False,
                "error": "本接口仅在本机开放（线上数据由定时任务自动更新）",
            }, 403)
        with _REFRESH_LOCK:
            already = _refresh["running"]
            if not already:
                _refresh.update({
                    "running": True, "started_at": time.time(), "finished_at": None,
                    "returncode": None, "iron_rc": None, "new_rows": None,
                    "error": None, "stage": None, "log": [],
                })
        if already:
            return json_resp(self, dict(_refresh_snapshot(), already=True))
        full = q.get("full", ["0"])[0] == "1"
        threading.Thread(target=_refresh_worker, args=(full,), daemon=True).start()
        return json_resp(self, {"ok": True, "started": True, "full": full})

    def handle_api(self, path, q):
        def arg(k, default=None):
            v = q.get(k)
            return v[0] if v else default

        if path == "/api/meta":
            return json_resp(self, {
                "ok": True,
                # 只暴露「已端到端接入」的品种（ENABLED_VARIETIES）。
                # 数据层认得但前端没接好的品种（如铁矿石）不出现在这里，
                # 否则前端会多出一个点进去是空表的选项。
                "varieties": {k: {"name": VARIETIES[k]["name"], "mult": VARIETIES[k]["mult"]}
                              for k in ENABLED_VARIETIES},
                "latest_date": ds.get_latest_date(),
                # 前端据此决定要不要显示「更新数据」按钮（线上为 false）
                "allow_refresh": ALLOW_REFRESH,
                # 当前服务进程 PID —— 本地调试用（页面页脚显示，便于确认"改的代码
                # 到底有没有生效 / 是不是起了两个服务"）。前端只在 allow_refresh
                # 为真（= 本机模式）时才显示，线上读者看不到。
                "pid": os.getpid(),
            })

        if path == "/api/refresh_status":
            if not ALLOW_REFRESH:
                return json_resp(self, {"ok": False, "error": "本接口仅在本机开放"}, 403)
            return json_resp(self, dict(_refresh_snapshot(), ok=True))

        if path == "/api/basis":
            variety = arg("variety", "IC")
            rows = ds.compute_basis_table(variety)
            spot = ds.load_spot(variety)
            spot_last = None
            if spot is not None and len(spot):
                spot_last = {"date": spot["date"].iloc[-1], "close": float(spot["spot_close"].iloc[-1])}
            # carry/波动：到期收益（基差率）÷ 近 20 日已实现波动率
            vol = engine.realized_vol(variety)
            for r in rows:
                r["carry_vol"] = engine.carry_per_vol(r.get("annual"), vol)
            return json_resp(self, {
                "ok": True, "variety": variety,
                "name": VARIETIES.get(variety, {}).get("name", ""),
                "spot": spot_last, "rows": rows,
                "vol": round(vol, 2) if vol is not None else None,
                "vol_window": 20,
            })

        # ---- 铁矿石图表（现货 / 最近合约 / 主力合约 / 基差率 / 年度对比）----
        # 数据不是抓来的，是**反推**的：现货 = 主力期货 ÷ (1 − 基差率/100)，
        # 基差率来自生意社 13 年逐日序列，期货来自本库。详见 scripts/build_ironore_series.py。
        if path == "/api/ironore":
            try:
                years = int(arg("years", "3"))
            except ValueError:
                years = 3
            return json_resp(self, ironore.payload(years))

        if path == "/api/realtime":
            variety = arg("variety", "IC")
            return json_resp(self, build_realtime_payload(variety))

        # ---- 批量实时：一次返回四个品种 ----
        # 为什么要这个接口：前端首屏原来并发打 4 个 /api/realtime，加上 /api/meta
        # 共 5 个请求。走 Cloudflare 回源时，只要任意一个请求卡住，
        # Promise.all 就整体挂起 → 页面永远转圈（2026-09-24 线上真实事故）。
        # 新浪接口本身支持逗号批量（见 realtime._fetch_raw），所以能一次性把所有
        # 品种的行情拉回来，四个品种共用一次出网请求。首屏请求数 5 → 2。
        if path == "/api/realtime_all":
            # 2026-11-14：改读后台聚合的内存快照。
            # 原实现每个 HTTP 请求串行发 5 次跨洋请求（16 期货代码 1 次 +
            # 4 个现货指数各 1 次，实测 2.51s），前端 5 秒轮询 ⇒ N 个用户
            # 就是 N×12 次/分外网抓取。现在外网频率与用户数彻底解耦。
            with _RT_LOCK:
                payload = _RT_CACHE.get("payload")
            if payload is None:
                payload = _build_realtime_all()   # 冷启动兜底：后台线程尚无首轮结果
            return json_resp(self, payload)

        if path == "/api/history":
            variety = arg("variety", "IC")
            years = int(arg("years", "1"))
            df = engine.variety_history(variety)
            if df is None or not len(df):
                return json_resp(self, {"ok": True, "variety": variety, "series": [], "stats": None})
            import pandas as pd
            if years > 0:
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
                sub = df[pd.to_datetime(df["date"]) >= cutoff]
                if len(sub) < 20:
                    sub = df
            else:
                sub = df
            cur = sub.iloc[-1]
            # 当前合约代号（次月合约），用于取实时价
            cur_symbol = cur["symbol"]

            # ---- 实时价覆盖"当前"这一点（跟基差看板/移仓助手口径一致）----
            live_annual = None
            live_basis = None
            live_spot = None
            live_date = None
            is_live = False
            try:
                cfg = VARIETIES.get(variety, {})
                spot_q = rt.fetch_index_quote(cfg.get("spot_code"))
                if spot_q and spot_q.get("price"):
                    qs = rt.fetch_futures_quote([cur_symbol])
                    if qs and qs[0].get("price"):
                        lp = qs[0]["price"]
                        sp = spot_q["price"]
                        ltd = ds.symbol_last_trade_date(cur_symbol)
                        from datetime import datetime as _dt
                        days_left = None
                        if ltd:
                            days_left = (_dt.strptime(ltd, "%Y-%m-%d").date() - _dt.now().date()).days
                        if days_left and days_left > 0:
                            live_spot = sp
                            live_basis = sp - lp
                            live_annual = (live_basis / sp) * 365 / days_left * 100
                            live_date = _dt.now().strftime("%Y-%m-%d")
                            is_live = True
            except Exception:
                pass

            ann_series = sub["annual"].dropna()
            cur_annual = float(cur["annual"]) if pd.notna(cur["annual"]) else None
            cur_basis = float(cur["basis"])

            series = [{"date": r["date"], "annual": r["annual"], "basis": r["basis"],
                       "spot": r["spot"], "symbol": r["symbol"]}
                      for _, r in sub.iterrows()]

            # ---- 三个指标各自算一套统计（前端切换指标时直接用，无需重新请求）----
            # 当前值优先用实时价
            hist_rows = series  # 不含实时点，作为分位基准
            live_cur = {}
            if is_live:
                live_cur = {"annual": live_annual, "basis": live_basis, "spot": live_spot}
                series.append({
                    "date": "今", "annual": round(live_annual, 2),
                    "basis": round(live_basis, 2), "spot": round(live_spot, 2),
                    "symbol": cur_symbol, "live": True,
                })

            metrics = {}
            for mk in ("annual", "basis", "spot"):
                st = engine.metric_stats(hist_rows, mk, current=live_cur.get(mk))
                if st:
                    metrics[mk] = st

            # ---- 固定期限口径（60 / 90 天）：与「滚动第 2 近月」并存的第二套 ----
            # 为什么需要：滚动第 2 近月的剩余天数在 1~70 天之间来回跳，年化 =
            # 基差率×365/天 的分母自带机械噪声，近月到期那几天天然占住历史高位，
            # 分位数被「离到期多远」污染。固定期限把分母钉死，同样一段历史才可比。
            # 详见 engine.fixed_tenor_history 的注释。
            tenors = {}
            # 当前波动率（近 20 日）—— 先算出来，下面滚动/固定期限两条 carry/波动 都要用
            vol = engine.realized_vol(variety)

            def _cv_pack(tenor_days, live_annual_v):
                """打包某个口径的 carry/波动 历史序列 + 统计。

                年化和波动率都按「当日」取，按日期内连接后再相除；
                用今日波动率去除历史年化是错的（等于拼两个不同时间的市场）。
                """
                cdf = engine.carry_vol_history(variety, tenor_days)
                if cdf is None or not len(cdf):
                    return None, None
                if years > 0:
                    csub = cdf[pd.to_datetime(cdf["date"]) >= cutoff]
                    if len(csub) < 20:
                        csub = cdf
                else:
                    csub = cdf
                rows = [{"date": r["date"], "annual": round(float(r["cv"]), 2)}
                        for _, r in csub.iterrows()]
                live_cv = engine.carry_per_vol(live_annual_v, vol)
                st = engine.metric_stats(rows, "annual", current=live_cv)
                if live_cv is not None:
                    rows = rows + [{"date": "今", "annual": round(live_cv, 2), "live": True}]
                return rows, st

            for T in engine.FT_TENORS:
                fdf = engine.fixed_tenor_history(variety, T)
                if fdf is None or not len(fdf):
                    continue
                if years > 0:
                    fsub = fdf[pd.to_datetime(fdf["date"]) >= cutoff]
                    if len(fsub) < 20:
                        fsub = fdf
                else:
                    fsub = fdf
                frows = [{"date": r["date"], "annual": r["annual"],
                          "basis_pct": r["basis_pct"],
                          "symbol": r["near_symbol"], "far_symbol": r["far_symbol"]}
                         for _, r in fsub.iterrows()]
                live_ft = _live_fixed_tenor(variety, T)
                st = engine.metric_stats(frows, "annual", current=live_ft)
                if not st:
                    continue
                if live_ft is not None:
                    frows = frows + [{"date": "今", "annual": round(live_ft, 2),
                                      "basis_pct": None, "symbol": None, "live": True}]
                tenors[str(T)] = {
                    "tenor": T, "series": frows, "metrics": st,
                    "live": live_ft is not None,
                    "live_current": round(live_ft, 2) if live_ft is not None else None,
                }
                # 该口径对应的 carry/波动 历史（分母钉死后的「性价比」序列）
                fcv_rows, fcv_st = _cv_pack(T, live_ft)
                tenors[str(T)]["cv_series"] = fcv_rows
                tenors[str(T)]["cv_metrics"] = fcv_st

            # carry/波动：给「年化贴水」配一个风险分母（详见 engine.realized_vol）
            a_cur = (metrics.get("annual") or {}).get("current")
            carry_vol = engine.carry_per_vol(a_cur, vol)
            # 滚动口径的 carry/波动 历史序列（前端第四种「指标」就是画它）
            cv_rows, cv_stats = _cv_pack(0, live_annual)

            # 兼容旧字段结构（前端已改为读 metrics，这里保留 cur_symbol 等）
            a_stats = metrics.get("annual") or {}
            return json_resp(self, {
                "ok": True, "variety": variety,
                "name": VARIETIES.get(variety, {}).get("name", ""),
                "series": series,
                "is_live": is_live,
                "is_trading": rt.is_trading_time(),
                "server_time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics,
                "tenors": tenors,
                "vol": round(vol, 2) if vol is not None else None,
                "vol_window": 20,
                "carry_vol": carry_vol,
                "cv_series": cv_rows,
                "cv_metrics": cv_stats,
                "stats": {
                    "cur_annual": a_stats.get("current"),
                    "cur_basis": (metrics.get("basis") or {}).get("current", cur_basis),
                    "cur_symbol": cur_symbol,
                    "percentile": a_stats.get("percentile"),
                    "q25": a_stats.get("q25"),
                    "q50": a_stats.get("q50"),
                    "q75": a_stats.get("q75"),
                    "sample": int(len(sub)),
                    "start": sub["date"].iloc[0], "end": sub["date"].iloc[-1],
                    "live_date": live_date,
                },
            })

        if path == "/api/move":
            variety = arg("variety", "IC")
            years = int(arg("years", "1"))
            # 尝试取实时价，用于计算"当前"价差与分位
            live_prices = None
            live_date = None
            try:
                # 「当前」价差按今天的在交易合约取，别再沿用数据最新日：
                # 否则摘牌合约的冻结价会混进来，算出 60%+ 的假年化。
                syms = _realtime_symbols(variety)
                qs = rt.fetch_futures_quote(syms)
                lp = {q["symbol"]: q["price"] for q in qs if q.get("price")}
                if lp:
                    live_prices = lp
                    live_date = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
            except Exception:
                pass
            # 合约对清单同样按今天构造（build_pairs 内部走 get_active_symbols），
            # 与 live_prices 同源，避免「9-30 的合约对 + 今天的价」口径打架。
            rows = engine.move_helper_table(variety, years=years, as_of=_today_str(),
                                            live_prices=live_prices, live_date=live_date)
            for r in rows:
                r.pop("series", None)
            return json_resp(self, {
                "ok": True, "variety": variety, "rows": rows,
                "is_trading": rt.is_trading_time(),
                "is_live": bool(live_prices),
                "server_time": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        if path == "/api/pair":
            near = arg("near")
            far = arg("far")
            years = int(arg("years", "1"))
            if not near or not far:
                return json_resp(self, {"ok": False, "error": "need near & far"}, 400)
            a = engine.pair_analysis(near, far, years=years)
            return json_resp(self, {"ok": True, "data": a})

        return json_resp(self, {"ok": False, "error": "unknown api"}, 404)


# ============================================================
# 实时行情后台聚合（2026-11-14）
#   20 个代码（16 期货合约 + 4 个现货指数）合并成 1 次新浪请求，实测 0.50s / 全命中；
#   后台单任务交易时段每 10s 抓一次放内存，handler 只读内存。
#   失败退避、非交易时段降频到 60s（保留最后一份收盘价，不空转刷屏）。
# ============================================================
_RT_LOCK = threading.Lock()
_RT_CACHE = {"payload": None, "ts": 0.0, "error": ""}
_RT_REFRESH_SEC = 10
_RT_IDLE_SEC = 60


def _build_realtime_all():
    varieties = list(ENABLED_VARIETIES)
    all_syms, sym_map, spot_codes = [], {}, []
    for v in varieties:
        syms = _realtime_symbols(v)
        sym_map[v] = syms
        all_syms.extend(syms)
        sc = VARIETIES.get(v, {}).get("spot_code")
        if sc:
            spot_codes.append(sc)
    spot_codes = list(dict.fromkeys(spot_codes))
    codes = ["nf_" + s.upper() for s in all_syms] + spot_codes
    raw = rt.fetch_bulk_raw(codes)                       # ← 唯一一次外网请求
    all_quotes = rt.fetch_futures_quote(all_syms, _raw=raw)
    spot_map = {sc: rt.fetch_index_quote(sc, _raw=raw) for sc in spot_codes}
    payloads = {}
    for v in varieties:
        mine = [q for q in all_quotes if q["symbol"] in set(sym_map[v])]
        payloads[v] = build_realtime_payload(
            v, quotes=mine,
            spot_q=spot_map.get(VARIETIES.get(v, {}).get("spot_code")))
    return {
        "ok": True,
        "varieties": payloads,
        "is_trading": rt.is_trading_time(),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _realtime_worker():
    failures = 0
    while True:
        try:
            payload = _build_realtime_all()
            with _RT_LOCK:
                _RT_CACHE["payload"] = payload
                _RT_CACHE["ts"] = time.time()
                _RT_CACHE["error"] = ""
            failures = 0
        except Exception as e:
            failures += 1
            with _RT_LOCK:
                _RT_CACHE["error"] = "%s %s" % (datetime.now().strftime("%H:%M:%S"), e)
            print("[realtime] 抓取失败(%d): %s" % (failures, e), flush=True)
        interval = _RT_REFRESH_SEC if rt.is_trading_time() else _RT_IDLE_SEC
        if failures:
            interval = min(60, interval * failures)
        time.sleep(interval)


def start_realtime_worker():
    t = threading.Thread(target=_realtime_worker, name="future-realtime", daemon=True)
    t.start()
    print("[realtime] 后台聚合已启动（交易 %ds / 非交易 %ds，单次 1 请求）"
          % (_RT_REFRESH_SEC, _RT_IDLE_SEC), flush=True)


def _warm_history_caches():
    """后台预热历史序列缓存（启动后立刻开始，顺序执行）。

    为什么需要：/api/history 第一次命中要现算 variety_history / 期限曲线 /
    固定期限两套 / 波动率与 carry-vol。ARM 上单个品种冷启动约 1.7s，
    四个品种并发打进来就是 4 份重活抢 GIL，首屏很容易撞上 20s 超时，
    表现是"有页面、图表空着"。改成启动就按品种顺序预热：
    用户点开时基本都是缓存命中（约 1.1s）。
    """
    time.sleep(1.0)  # 先让 HTTP 端口和实时快照起来，别和高频行情抢
    for code in ENABLED_VARIETIES:
        try:
            engine.variety_history(code)
            engine.daily_contract_curves(code)
            for T in engine.FT_TENORS:
                engine.fixed_tenor_history(code, T)
            engine.vol_history(code)
            for T in (0,) + tuple(engine.FT_TENORS):
                engine.carry_vol_history(code, T)
            print(f"[warm] {code} 历史缓存就绪", flush=True)
        except Exception as e:  # 预热失败不能影响服务
            print(f"[warm] {code} 预热失败: {e}", flush=True)


def start_warm_worker():
    t = threading.Thread(target=_warm_history_caches, name="future-warm", daemon=True)
    t.start()


def main():
    start_realtime_worker()
    start_warm_worker()
    port = int(os.environ.get("FUTURE_PORT", DEFAULT_PORT))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"future 服务已启动: http://127.0.0.1:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
