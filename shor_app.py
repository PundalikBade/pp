"""Shor Lab web app: type a number, run Shor's algorithm on a simulator, BlueQubit, or real IBM
quantum hardware, and get the factors back in the page.

    pip install qiskit qiskit-aer bluequbit qiskit-ibm-runtime
    python shor_app.py            -> http://localhost:8000

Public hosting (Render, Docker): set SHOR_PUBLIC=1. Every visitor then brings their own API key;
keys are used for that one request and never stored, logged, or shared with other visitors.
"""

import contextlib
import io
import json
import os
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import shor_bluequbit as shor

PUBLIC = os.environ.get("SHOR_PUBLIC") == "1"
JOBS: dict[str, dict] = {}
SIM_LOCK = threading.Lock()  # one local simulation at a time (they are CPU heavy)
MAX_N = {"local": 4095, "cpu": 4095, "gpu": 4095, "ibm-fake": 21, "ibm": 63, "quantum": 35}


class _Tee(io.TextIOBase):
    def __init__(self, job):
        self.job = job

    def write(self, s):
        self.job["log"] += s
        return len(s)


_orig_run_circuit = shor.run_circuit


def _recording_run_circuit(qc, device, shots):
    counts = _orig_run_circuit(qc, device, shots)
    job = getattr(threading.current_thread(), "job", None)
    if job is not None:
        job["counts"] = {k.replace(" ", ""): int(v) for k, v in counts.items()}
    return counts


shor.run_circuit = _recording_run_circuit


def _start(target, params):
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "status": "running", "log": "", "started": time.time()}
    JOBS[jid] = job

    def wrapper():
        threading.current_thread().job = job
        shor.KEYS.bluequbit = params.get("bq_token") or None
        shor.KEYS.ibm = params.get("ibm_token") or None
        try:
            with contextlib.redirect_stdout(_Tee(job)):
                job.update(status="done", **target(params))
        except Exception as e:
            job["log"] += "\n" + "".join(traceback.format_exception_only(type(e), e))
            job.update(status="error", error=str(e) or type(e).__name__)
        finally:
            shor.KEYS.ibm = shor.KEYS.bluequbit = None
            job["elapsed"] = round(time.time() - job["started"], 1)

    threading.Thread(target=wrapper, daemon=True).start()
    return jid


def _factor(params):
    if not SIM_LOCK.acquire(timeout=600):
        raise RuntimeError("The server is busy with another run. Try again in a minute.")
    try:
        N = int(params["N"])
        print(f"N = {N} ({N.bit_length()} bits), device = {params['device']}")
        f = shor.factorize(N, device=params["device"], shots=int(params.get("shots") or 1024),
                           a=int(params["a"]) if params.get("a") else None,
                           tries=1 if params["device"] == "quantum" else 6)
        print(f"\nRESULT: {N} = {' x '.join(map(str, f))}")
        return {"factors": [str(x) for x in f]}
    finally:
        SIM_LOCK.release()


def _ibm_submit(params):
    return {"ibm": shor.ibm_submit(int(params["N"]), int(params.get("shots") or 1000))}


def _batch(params):
    if not SIM_LOCK.acquire(timeout=600):
        raise RuntimeError("The server is busy with another run. Try again in a minute.")
    try:
        return {"batch": shor.ibm_batch(params["Ns"], params["device"], int(params.get("shots") or 1000))}
    finally:
        SIM_LOCK.release()


def _with_ibm_key(params, fn):
    shor.KEYS.ibm = params.get("ibm_token") or None
    try:
        return 200, fn()
    except Exception as e:
        return 400, {"error": str(e) or type(e).__name__}
    finally:
        shor.KEYS.ibm = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = json.dumps(body) if not isinstance(body, (str, bytes)) else body
        data = data.encode() if isinstance(data, str) else data
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass  # never log requests: bodies carry API keys

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/api/config":
            return self._send(200, {"public": PUBLIC,
                                    "bluequbit_key": not PUBLIC and bool(os.environ.get("BLUEQUBIT_API_TOKEN")),
                                    "ibm_key": not PUBLIC and bool(os.environ.get("IBM_QUANTUM_TOKEN"))})
        if self.path.startswith("/api/job/"):
            job = JOBS.get(self.path.rsplit("/", 1)[-1])
            if not job:
                return self._send(404, {"error": "This run is no longer on the server (it restarted). Start it again."})
            return self._send(200, dict(job, elapsed=job.get("elapsed") or round(time.time() - job["started"], 1)))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._send(400, {"error": "Bad request."})
        try:
            if self.path == "/api/run":
                d = p.get("device")
                assert d in MAX_N, "Pick a device."
                N = int(p.get("N") or 0)
                assert 4 <= N <= MAX_N[d], f"For this device enter N from 4 to {MAX_N[d]}."
                if d in ("ibm", "quantum"):
                    assert p.get("confirm"), "Tick the box to confirm a real hardware run."
                if d == "ibm":
                    return self._send(200, {"id": _start(_ibm_submit, p)})
                return self._send(200, {"id": _start(_factor, p)})
            if self.path == "/api/ibm-check":
                code, body = _with_ibm_key(
                    p, lambda: shor.ibm_check(str(p["job_id"]), int(p["N"]), int(p["a"]), int(p["t"])))
                return self._send(code, body)
            if self.path == "/api/ibm-usage":
                code, body = _with_ibm_key(p, shor.ibm_usage)
                return self._send(code, body)
            if self.path == "/api/batch":
                Ns = sorted({int(x) for x in str(p.get("Ns", "")).replace(",", " ").split()})
                assert Ns, "Enter one or more numbers, e.g. 15, 21, 33, 35."
                assert p.get("device") in ("ibm", "ibm-fake"), "Pick an IBM device."
                assert len(Ns) <= 12 and max(Ns) <= MAX_N[p["device"]], \
                    f"Use up to 12 numbers, each at most {MAX_N[p['device']]}."
                if p["device"] == "ibm":
                    assert p.get("confirm"), "Tick the box to confirm a real hardware run."
                p["Ns"] = Ns
                return self._send(200, {"id": _start(_batch, p)})
        except (ValueError, KeyError, AssertionError) as e:
            return self._send(400, {"error": str(e) or "Check the numbers you entered."})
        self._send(404, {"error": "not found"})


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shor Lab Hardware</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap">
<style>
:root{--bg:#F2F4F7;--panel:#fff;--line:#D5DBE4;--fg:#15202C;--muted:#5A6878;--accent:#2C55D6;--ink:#fff;--peak:#C9780F;--ok:#18794E;--bad:#B42318;--code:#EEF1F6;
--d:"Space Grotesk",system-ui,sans-serif;--b:"IBM Plex Sans",system-ui,sans-serif;--m:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0D1218;--panel:#151C25;--line:#28323F;--fg:#DCE3EB;--muted:#93A1B1;--accent:#7D9CFF;--ink:#0D1218;--peak:#F0AE47;--ok:#4CC38A;--bad:#F2716A;--code:#1B2430;color-scheme:dark}}
[hidden]{display:none!important}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 var(--b)}
.wrap{max-width:1120px;margin:0 auto;padding:28px 16px 60px;display:grid;gap:24px}.wrap>*{min-width:0}
h1,h2,h3{font-family:var(--d);margin:0;line-height:1.15}h1{font-size:clamp(28px,4vw,38px)}h2{font-size:20px}h3{font-size:16px}
p{margin:0;max-width:70ch}.lede{color:var(--muted)}
.eyebrow{font:12px var(--m);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.grid{display:grid;grid-template-columns:350px minmax(0,1fr);gap:20px;align-items:start}
@media(max-width:860px){.grid{grid-template-columns:minmax(0,1fr)}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;display:grid;gap:12px;min-width:0}
label{font-size:13px;font-weight:600;color:var(--muted)}.field{display:grid;gap:5px;min-width:0}
input,select{font:500 15px var(--m);background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:9px 10px;width:100%}
input:focus-visible,select:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#N{font-size:22px;font-weight:600;padding:12px}
.row{display:flex;gap:10px;flex-wrap:wrap}.row>*{flex:1 1 120px}
.presets{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:500 12px var(--m);border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:999px;padding:4px 10px;cursor:pointer}
.btn{font:600 16px var(--b);border:0;border-radius:8px;padding:12px;background:var(--accent);color:var(--ink);cursor:pointer}.btn:disabled{opacity:.5;cursor:default}
.btn.ghost{background:transparent;color:var(--fg);border:1px solid var(--line);font-size:14px}
.warn{border:1px solid var(--peak);border-radius:8px;padding:10px;font-size:13px;display:grid;gap:6px}
.chk{display:flex;gap:8px;align-items:flex-start;font-weight:400;color:var(--fg)}.chk input{width:auto;margin-top:3px}
.hint{font-size:13px;color:var(--muted);margin:0}
.res{font:600 clamp(20px,3vw,28px) var(--m);overflow-wrap:anywhere;display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline}
.res .f{color:var(--ok)}.res .e{color:var(--muted);font-weight:400;font-size:16px}.res .x{color:var(--bad);font-size:18px}
.status{font:13px var(--m);color:var(--muted);overflow-wrap:anywhere}
.steps{display:flex;flex-wrap:wrap;gap:6px}.step{font:600 11px var(--m);letter-spacing:.04em;text-transform:uppercase;padding:4px 9px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.step.on{border-color:var(--accent);color:var(--accent)}.step.done{border-color:var(--ok);color:var(--ok)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.fact{border:1px solid var(--line);border-radius:8px;padding:10px;display:grid;gap:2px;min-width:0}
.fact b{font:600 17px var(--m);overflow-wrap:anywhere}.fact span{font-size:12px;color:var(--muted)}
pre{font:12.5px/1.55 var(--m);background:var(--code);border-radius:8px;padding:12px;max-height:300px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}
.bars{display:grid;gap:5px;font:12px var(--m);font-variant-numeric:tabular-nums}.bar{display:grid;grid-template-columns:118px minmax(0,1fr) 50px;gap:8px;align-items:center}
.bar .t{height:13px;background:var(--code);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--peak)}.bar.ok i{background:var(--ok)}.bar .v{text-align:right}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--muted)}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:640px;font-size:14px}
th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted)}td{font-family:var(--m)}
.good{color:var(--ok)}.badc{color:var(--bad)}
</style></head><body><div class="wrap">
<header style="display:grid;gap:8px"><div class="eyebrow">Shor's algorithm · real quantum hardware</div><h1>Shor Lab Hardware</h1>
<p class="lede">Type a number. It is sent with your API key to a simulator or a real IBM quantum chip, the period is measured, and the factors come back here.</p></header>
<div class="grid">
<form class="panel" id="f" autocomplete="off">
 <div class="field"><label for="N">Number to factor</label><input id="N" value="15" inputmode="numeric">
  <div class="presets"><button type="button" class="chip" data-n="15">15</button><button type="button" class="chip" data-n="21">21</button><button type="button" class="chip" data-n="33">33</button><button type="button" class="chip" data-n="35">35</button><button type="button" class="chip" data-n="143">143</button></div></div>
 <div class="field"><label for="device">Where to run</label><select id="device">
  <option value="ibm">IBM quantum hardware (real chip)</option>
  <option value="ibm-fake">IBM chip noise model (free, simulated)</option>
  <option value="local">Simulator on this server (free)</option>
  <option value="cpu">BlueQubit CPU simulator (free, needs key)</option>
  <option value="gpu">BlueQubit GPU simulator (needs credit)</option>
  <option value="quantum">BlueQubit quantum hardware (needs credit)</option></select></div>
 <div class="field" id="ibmk"><label for="ibm_token">IBM Quantum API key</label><input id="ibm_token" type="password" placeholder="Paste your IBM API key"></div>
 <div class="field" id="bqk" hidden><label for="bq_token">BlueQubit API token</label><input id="bq_token" type="password" placeholder="Paste your BlueQubit token"></div>
 <div class="row"><div class="field"><label for="shots">Shots</label><input id="shots" value="1000" inputmode="numeric"></div><div class="field" id="af"><label for="a">Base a</label><input id="a" placeholder="auto" inputmode="numeric"></div></div>
 <div class="warn" id="hw"><b>Real quantum hardware</b><span>Sends one job. It uses a few seconds of your free IBM minutes and can wait in IBM's queue for minutes or hours. You can close this page and come back: the job is remembered on this device.</span>
  <label class="chk"><input type="checkbox" id="confirm"> Use my hardware time for this run</label></div>
 <button class="btn" id="go">Factor on IBM quantum hardware</button>
 <button class="btn ghost" id="use" type="button">Check my IBM minutes left</button>
 <div class="status" id="status" role="status"></div>
 <p class="hint">Your key is sent only with each request, to IBM or BlueQubit. It is never saved on the server or in this page.</p>
</form>
<div class="panel" aria-live="polite">
 <h2>Result</h2>
 <div class="steps" id="steps"></div>
 <div class="res" id="res"><span class="e">Pick a number and press the button.</span></div>
 <div class="facts" id="facts"></div>
 <div id="chart" hidden style="display:grid;gap:8px"><h3>Measured outcomes</h3><div class="bars" id="bars"></div>
  <div class="legend"><span><i style="background:var(--ok)"></i>correct peak (k·2^t/r)</span><span><i style="background:var(--peak)"></i>other outcome (noise)</span></div></div>
 <h3>Log</h3><pre id="log"></pre>
</div></div>
<section class="panel" id="batch">
 <h2>Batch test on IBM</h2>
 <p class="hint">Several numbers in one IBM job: one queue wait. Keep this page open until IBM finishes.</p>
 <div class="row"><div class="field"><label for="bNs">Numbers</label><input id="bNs" value="15, 21, 33, 35"></div>
  <div class="field"><label for="bDev">Device</label><select id="bDev"><option value="ibm">IBM quantum hardware</option><option value="ibm-fake">IBM noise model (free, N ≤ 21)</option></select></div></div>
 <label class="chk" id="bConfF"><input type="checkbox" id="bConf"> Use my hardware time for this batch</label>
 <button class="btn" id="bGo" type="button">Run batch</button>
 <div class="status" id="bStatus"></div>
 <div class="tw"><table id="bTable" hidden><thead><tr><th>N</th><th>a</th><th>Qubits</th><th>2-qubit gates</th><th>Period</th><th>Shots on correct peaks</th><th>Random would give</th><th>Factors</th></tr></thead><tbody></tbody></table></div>
 <pre id="bLog" hidden></pre>
</section>
</div>
<script>
const $=id=>document.getElementById(id);let cfg={};
const store={get(){try{return JSON.parse(localStorage.getItem('ibmJob')||'null')}catch(e){return null}},set(v){try{v?localStorage.setItem('ibmJob',JSON.stringify(v)):localStorage.removeItem('ibmJob')}catch(e){}}};
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(async r=>({ok:r.ok,j:await r.json()}));
const key=()=>$('ibm_token').value.trim();
function sync(){const d=$('device').value,hw=d==='ibm'||d==='quantum';$('hw').hidden=!hw;
 $('ibmk').hidden=!((d==='ibm'||$('bDev').value==='ibm')&&!cfg.ibm_key);$('bqk').hidden=!(['cpu','gpu','quantum'].includes(d)&&!cfg.bluequbit_key);
 $('af').hidden=d==='ibm';$('go').textContent=d==='ibm'?'Factor on IBM quantum hardware':hw?'Factor on quantum hardware':'Factor N';$('bConfF').hidden=$('bDev').value!=='ibm'}
$('device').onchange=sync;$('bDev').onchange=sync;
document.querySelectorAll('[data-n]').forEach(b=>b.onclick=()=>{$('N').value=b.dataset.n});
fetch('/api/config').then(r=>r.json()).then(c=>{cfg=c;sync()}).catch(sync);
const STEPS=['Submitted','In IBM queue','Running on chip','Result'];
function steps(i){const s=$('steps');s.textContent='';STEPS.forEach((t,k)=>{const e=document.createElement('span');e.className='step'+(k<i?' done':k===i?' on':'');e.textContent=t;s.append(e)})}
function say(t,cls){const r=$('res');r.textContent='';const s=document.createElement('span');s.className=cls||'e';s.textContent=t;r.append(s)}
function showFactors(N,f){const r=$('res');r.textContent='';const n=document.createElement('span');n.textContent=N+' =';r.append(n);f.forEach((x,i)=>{if(i){const t=document.createElement('span');t.className='e';t.textContent='×';r.append(t)}const s=document.createElement('span');s.className='f';s.textContent=x;r.append(s)})}
function facts(list){const b=$('facts');b.textContent='';list.forEach(([v,l])=>{const d=document.createElement('div');d.className='fact';const x=document.createElement('b');x.textContent=v;const y=document.createElement('span');y.textContent=l;d.append(x,y);b.append(d)})}
function bars(top,N,a,t){if(!top||!top.length){$('chart').hidden=true;return}
 const Q=2**t;let r=1,x=a%N;while(x!==1){x=x*a%N;r++}const peaks=new Set();for(let k=0;k<r;k++){const c=k*Q/r;peaks.add(Math.floor(c)%Q);peaks.add(Math.ceil(c)%Q)}
 const max=top[0][2],box=$('bars');box.textContent='';
 top.forEach(([bits,y,c])=>{const row=document.createElement('div');row.className='bar'+(peaks.has(y)?' ok':'');const l=document.createElement('span');l.textContent=bits.slice(-10)+' ('+y+')';
  const tr=document.createElement('span');tr.className='t';const i=document.createElement('i');i.style.width=(100*c/max)+'%';tr.append(i);const v=document.createElement('span');v.className='v';v.textContent=c;row.append(l,tr,v);box.append(row)});$('chart').hidden=false}
function busy(b){$('go').disabled=b}
async function pollLocal(id,onDone){const r=await fetch('/api/job/'+id);const s=await r.json();if(!r.ok){busy(false);say(s.error,'x');return}
 $('log').textContent=s.log||'';$('log').scrollTop=1e9;
 if(s.status==='running'){$('status').textContent='Working… '+s.elapsed+' s';return setTimeout(()=>pollLocal(id,onDone),1500)}onDone(s)}
let checkTimer=null;
async function checkIBM(job){clearTimeout(checkTimer);if(!key()&&!cfg.ibm_key){busy(false);steps(1);$('status').textContent='IBM job '+job.job_id+' is saved. Paste your IBM key, then press the button to check it.';$('go').textContent='Check my IBM job';$('go').dataset.check='1';return}
 const {ok,j}=await post('/api/ibm-check',{job_id:job.job_id,N:job.N,a:job.a,t:job.t,ibm_token:key()});
 if(!ok){$('status').textContent='Could not check the job: '+j.error;busy(false);return}
 const st=j.status;steps(st==='QUEUED'||st==='INITIALIZING'?1:st==='RUNNING'?2:st==='DONE'?4:1);
 $('status').textContent='IBM job '+job.job_id+' on '+job.backend+': '+st+' · checked '+new Date().toLocaleTimeString();
 if(st==='DONE'){busy(false);store.set(null);delete $('go').dataset.check;sync();
  if(j.factors)showFactors(job.N,j.factors.map(String));else say('No factors this time: noise hid the period. Try again or use a smaller N.','x');
  facts([[j.r_found??'none','Period found (expected '+job.r_expected+')'],[(100*j.peak_share).toFixed(1)+'%','Shots on correct peaks (perfect: ~100%)'],[(100*j.noise_share).toFixed(1)+'%','Random noise would give'],[job.two_qubit_gates,'Two-qubit gates on '+job.backend],[j.quantum_seconds!=null?j.quantum_seconds+' s':'—','IBM time used']]);
  bars(j.top,job.N,job.a,job.t);return}
 if(st==='ERROR'||st==='CANCELLED'){busy(false);store.set(null);say('IBM reported '+st+(j.error?': '+j.error:''),'x');return}
 checkTimer=setTimeout(()=>checkIBM(job),window.__fast?500:20000)}
$('f').onsubmit=async ev=>{ev.preventDefault();
 if($('go').dataset.check){const s=store.get();if(s){busy(true);return checkIBM(s)}}
 const d=$('device').value;const body={N:$('N').value.trim(),a:$('a').value.trim(),shots:$('shots').value,device:d,confirm:$('confirm').checked,ibm_token:key(),bq_token:$('bq_token').value.trim()};
 if(d==='ibm'&&!body.ibm_token&&!cfg.ibm_key){$('status').textContent='Paste your IBM API key first.';return}
 busy(true);$('log').textContent='';$('facts').textContent='';$('chart').hidden=true;$('steps').textContent='';$('status').textContent='';say('Working on '+body.N+'…');
 const {ok,j}=await post('/api/run',body);if(!ok){busy(false);say(j.error,'x');return}
 if(d==='ibm'){steps(0);$('status').textContent='Compiling the circuit for the IBM chip…';
  pollLocal(j.id,s=>{if(s.status!=='done'){busy(false);say('Could not submit: '+s.error,'x');return}const job=s.ibm;store.set(job);say('Sent to IBM. Waiting for the chip…');
   facts([[job.backend,'IBM chip'],[job.qubits,'Qubits'],[job.two_qubit_gates,'Two-qubit gates'],[job.job_id,'IBM job id']]);checkIBM(job)});return}
 pollLocal(j.id,s=>{busy(false);$('status').textContent='Finished in '+s.elapsed+' s';if(s.status==='done'){showFactors(body.N,s.factors);
  if(s.counts){const top=Object.entries(s.counts).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([b,c])=>[b,parseInt(b,2),c]);const m=s.log.match(/a = (\d+)/g);const a=m?+m[m.length-1].split('= ')[1]:0;if(a)bars(top,+body.N,a,top[0][0].length)}}
  else say('Failed: '+s.error,'x')})};
$('use').onclick=async()=>{$('status').textContent='Checking…';const {ok,j}=await post('/api/ibm-usage',{ibm_token:key()});
 if(!ok){$('status').textContent='Could not read your IBM usage: '+j.error;return}
 const lim=j.usage_limit_seconds??j.usage_allocation_seconds,used=j.usage_consumed_seconds??0,left=j.usage_remaining_seconds??(lim!=null?lim-used:null);
 $('status').textContent='IBM time used: '+used+' s'+(lim!=null?' of '+lim+' s · left: '+left+' s ('+(left/60).toFixed(1)+' min)':'')};
$('bGo').onclick=async()=>{$('bGo').disabled=true;const body={ibm_token:key(),Ns:$('bNs').value,device:$('bDev').value,confirm:$('bConf').checked};
 $('bStatus').textContent='Building circuits…';$('bLog').hidden=false;$('bLog').textContent='';$('bTable').hidden=true;
 const {ok,j}=await post('/api/batch',body);if(!ok){$('bGo').disabled=false;$('bStatus').textContent=j.error;return}
 const poll=async()=>{const r=await fetch('/api/job/'+j.id);const s=await r.json();$('bLog').textContent=s.log||'';$('bLog').scrollTop=1e9;
  if(s.status==='running'){$('bStatus').textContent='Running… '+s.elapsed+' s';return setTimeout(poll,2500)}$('bGo').disabled=false;
  if(s.status!=='done'){$('bStatus').textContent='Failed: '+(s.error||'server restarted');return}
  $('bStatus').textContent='Finished in '+s.elapsed+' s'+(s.batch.quantum_seconds!=null?' · IBM time used: '+s.batch.quantum_seconds+' s':'');
  const tb=$('bTable').querySelector('tbody');tb.textContent='';
  s.batch.rows.forEach(x=>{const tr=document.createElement('tr');const strong=x.peak_share>5*x.noise_share;
   [x.N,x.a,x.qubits,x.two_qubit_gates,(x.r_found??'none')+' (exp. '+x.r_expected+')',(100*x.peak_share).toFixed(1)+'%',(100*x.noise_share).toFixed(1)+'%',x.factors?x.factors.join(' × '):'not found']
   .forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===5)td.className=strong?'good':'badc';if(i===7)td.className=x.factors?'good':'badc';tr.append(td)});tb.append(tr)});$('bTable').hidden=false};poll()};
const saved=store.get();if(saved){$('N').value=saved.N;say('You have an IBM job waiting for N = '+saved.N+'.');facts([[saved.backend,'IBM chip'],[saved.job_id,'IBM job id']]);steps(1);busy(true);setTimeout(()=>checkIBM(saved),300)}
</script></body></html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if PUBLIC or os.environ.get("PORT") else "127.0.0.1"
    print(f"Shor Lab running at http://localhost:{port}  (Ctrl+C to stop){'  [public mode]' if PUBLIC else ''}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
