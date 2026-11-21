"""
Health insurance cost estimator: Flask app serving XGBRegressor.pkl

Model inputs (in training order): age, sex, bmi, children, smoker, region
Encoding (read from the model's own splits): sex female=0 male=1,
smoker no=0 yes=1, region northeast=0 northwest=1 southeast=2 southwest=3.

Local run:   python app.py
Render run:  gunicorn app:app
"""
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("estimator")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / os.environ.get("MODEL_FILE", "XGBRegressor.pkl")

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
SEX = {"female": 0, "male": 1}
SMOKER = {"no": 0, "yes": 1}
REGION = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
LABELS = {
    "age": "Age", "sex": "Sex", "bmi": "BMI", "children": "Children",
    "smoker": "Smoking", "region": "Region",
}

with open(MODEL_PATH, "rb") as fh:
    model = pickle.load(fh)
log.info("Model loaded from %s", MODEL_PATH)

try:
    _imp = [float(x) for x in model.feature_importances_]
    IMPORTANCE = sorted(
        ({"name": LABELS[f], "value": round(v * 100, 1)} for f, v in zip(FEATURES, _imp)),
        key=lambda d: d["value"], reverse=True,
    )
except Exception:  # importance is decorative; never block startup
    IMPORTANCE = []

app = Flask(__name__)


def parse_payload(data):
    """Validate the request body and return a one-row DataFrame."""
    try:
        age = int(data["age"])
        bmi = float(data["bmi"])
        children = int(data["children"])
        sex = SEX[str(data["sex"]).lower()]
        smoker = SMOKER[str(data["smoker"]).lower()]
        region = REGION[str(data["region"]).lower()]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Please complete every field with valid values.")

    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100.")
    if not 10 <= bmi <= 60:
        raise ValueError("BMI must be between 10 and 60.")
    if not 0 <= children <= 10:
        raise ValueError("Children must be between 0 and 10.")

    row = pd.DataFrame(
        [[age, sex, bmi, children, smoker, region]], columns=FEATURES
    ).astype({"age": "int64", "sex": "int64", "bmi": "float64",
              "children": "int64", "smoker": "int64", "region": "int64"})
    return row


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        row = parse_payload(data)
        charges = max(float(model.predict(row)[0]), 0.0)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="The model could not produce an estimate. Try again."), 500
    return jsonify(charges=round(charges, 2), monthly=round(charges / 12, 2))


@app.get("/health")
def health():
    return jsonify(status="ok")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cover cost estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --abyss:#0B3A42; --abyss-2:#0F4A53; --lagoon:#137C86; --aqua:#4FD1C5;
  --mist:#EEF5F5; --paper:#FBFDFD; --ink:#11292E; --muted:#5C7479;
  --amber:#F2A93B; --alert:#C8372D; --line:#D6E3E4;
  --head:'Sora',system-ui,sans-serif; --body:'Manrope',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0}
html{-webkit-text-size-adjust:100%}
body{font-family:var(--body);color:var(--ink);background:var(--paper);min-height:100vh;
  display:grid;grid-template-columns:minmax(340px,5fr) 6fr;line-height:1.55}

/* ---------- left stage ---------- */
.stage{background:radial-gradient(120% 90% at 0% 0%,var(--abyss-2) 0%,var(--abyss) 60%);
  color:#E8F4F4;padding:44px 48px;display:flex;flex-direction:column;gap:28px;
  position:sticky;top:0;height:100vh;overflow:hidden}
.brand{display:flex;align-items:center;gap:10px;font-family:var(--head);font-weight:600;font-size:1.05rem}
.brand i{width:30px;height:30px;border-radius:9px;background:var(--aqua);display:grid;place-items:center}
.brand svg{width:18px;height:18px}
.stage h1{font-family:var(--head);font-weight:600;font-size:clamp(1.7rem,2.6vw,2.4rem);line-height:1.18;max-width:15ch;letter-spacing:-.01em}
.stage p.lede{color:#A9C9CC;max-width:38ch;font-size:.97rem}

.pulse{width:100%;height:84px;margin:-4px 0}
.pulse .track{stroke:rgba(255,255,255,.1)}
.pulse .beat{stroke:var(--aqua);stroke-dasharray:170 830;animation:sweep 3.4s linear infinite;
  filter:drop-shadow(0 0 6px rgba(79,209,197,.7))}
@keyframes sweep{from{stroke-dashoffset:1000}to{stroke-dashoffset:0}}

.result{display:grid;grid-template-columns:auto 1fr;gap:22px;align-items:center}
.ring{position:relative;width:148px;height:148px}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:10;stroke-linecap:round}
.ring .bg{stroke:rgba(255,255,255,.12)}
.ring .fg{stroke:var(--amber);stroke-dasharray:440;stroke-dashoffset:440;
  transition:stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1)}
.ring b{position:absolute;inset:0;display:grid;place-items:center;font-family:var(--head);font-size:.8rem;
  color:#A9C9CC;font-weight:500;text-align:center;padding:0 22px}
.figure small{display:block;color:#A9C9CC;font-size:.85rem}
.figure output{display:block;font-family:var(--head);font-weight:700;font-size:clamp(2rem,3.4vw,2.9rem);
  letter-spacing:-.02em;line-height:1.1;margin:4px 0 6px;font-variant-numeric:tabular-nums}
.figure .sub{color:var(--aqua);font-weight:600;font-size:.92rem;min-height:1.4em}

.weights{margin-top:auto}
.weights h2{font-family:var(--head);font-weight:500;font-size:.92rem;color:#A9C9CC;margin-bottom:12px}
.w{display:grid;grid-template-columns:78px 1fr 44px;align-items:center;gap:10px;font-size:.85rem;margin-bottom:8px}
.w span:last-child{text-align:right;color:#A9C9CC;font-variant-numeric:tabular-nums}
.w div{height:6px;border-radius:6px;background:rgba(255,255,255,.1);overflow:hidden}
.w em{display:block;height:100%;width:0;background:var(--aqua);border-radius:6px;transition:width 1s cubic-bezier(.2,.8,.2,1)}

/* ---------- right form ---------- */
main{padding:56px clamp(24px,5vw,72px);display:flex;align-items:center;justify-content:center}
form{width:100%;max-width:560px}
form>h2{font-family:var(--head);font-weight:600;font-size:1.5rem;letter-spacing:-.01em}
form>p{color:var(--muted);margin:6px 0 30px}
.field{margin-bottom:24px}
.field>label,.field>legend{display:block;font-weight:600;font-size:.92rem;margin-bottom:9px;padding:0}
fieldset{border:0;padding:0;min-width:0}
.row{display:flex;align-items:center;gap:16px}
.hint{color:var(--muted);font-size:.84rem;margin-top:7px}

input[type=number]{font:inherit;color:var(--ink);background:#fff;border:1.5px solid var(--line);
  border-radius:10px;padding:11px 14px;width:104px;transition:border-color .15s,box-shadow .15s}
input[type=number]:focus{outline:none;border-color:var(--lagoon);box-shadow:0 0 0 4px rgba(19,124,134,.14)}
input[type=range]{flex:1;accent-color:var(--lagoon);height:30px;cursor:pointer}

.seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;background:var(--mist);border-radius:12px;padding:4px;gap:4px}
.seg.four{grid-auto-flow:row;grid-template-columns:repeat(4,1fr)}
.seg label{position:relative;cursor:pointer}
.seg input{position:absolute;opacity:0;inset:0}
.seg span{display:block;text-align:center;padding:10px 8px;border-radius:9px;font-weight:600;font-size:.92rem;
  color:var(--muted);transition:background .18s,color .18s,box-shadow .18s}
.seg input:checked+span{background:#fff;color:var(--abyss);box-shadow:0 1px 3px rgba(11,58,66,.18)}
.seg input:focus-visible+span{outline:3px solid rgba(19,124,134,.45);outline-offset:1px}
.seg.smoke input[value=yes]:checked+span{background:var(--abyss);color:#fff}

.chip{display:inline-block;font-size:.8rem;font-weight:700;padding:4px 11px;border-radius:99px;
  background:#E3F3F1;color:#0B6B63;transition:background .25s,color .25s}
.chip.warn{background:#FCEFD6;color:#8A5A00}
.chip.high{background:#FBE0DD;color:#9C2A21}
.chip.low{background:#E2ECF8;color:#25508A}

details{margin-top:10px}
summary{cursor:pointer;color:var(--lagoon);font-weight:600;font-size:.88rem;width:fit-content}
details .row{margin-top:10px;gap:10px;flex-wrap:wrap}
details input{width:112px}
details button{font:inherit;font-weight:600;font-size:.88rem;border:1.5px solid var(--lagoon);color:var(--lagoon);
  background:transparent;border-radius:10px;padding:10px 14px;cursor:pointer}
details button:hover{background:rgba(19,124,134,.08)}

.step{display:inline-flex;align-items:center;border:1.5px solid var(--line);border-radius:10px;background:#fff;overflow:hidden}
.step button{width:42px;height:42px;border:0;background:transparent;font-size:1.25rem;color:var(--lagoon);cursor:pointer}
.step button:hover{background:var(--mist)}
.step output{min-width:44px;text-align:center;font-weight:700;font-variant-numeric:tabular-nums}

.go{width:100%;font:inherit;font-family:var(--head);font-weight:600;font-size:1rem;color:#fff;
  background:var(--lagoon);border:0;border-radius:12px;padding:16px;cursor:pointer;margin-top:6px;
  transition:background .2s,transform .12s;display:flex;justify-content:center;align-items:center;gap:10px}
.go:hover{background:var(--abyss-2)}
.go:active{transform:scale(.985)}
.go:focus-visible{outline:3px solid rgba(19,124,134,.45);outline-offset:3px}
.go[disabled]{opacity:.75;cursor:progress}
.spin{width:16px;height:16px;border:2.5px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;
  animation:rot .7s linear infinite;display:none}
.go[disabled] .spin{display:block}
@keyframes rot{to{transform:rotate(360deg)}}
.err{color:var(--alert);font-weight:600;font-size:.9rem;min-height:1.3em;margin-top:12px}
.fine{color:var(--muted);font-size:.8rem;margin-top:14px}

@media (max-width:900px){
  body{grid-template-columns:1fr}
  .stage{position:static;height:auto;padding:32px 24px}
  .weights{margin-top:6px}
  main{padding:36px 24px 56px}
}
@media (max-width:420px){.seg.four{grid-template-columns:repeat(2,1fr)}.result{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){
  .pulse .beat{animation:none;stroke-dasharray:none}
  .ring .fg,.w em{transition:none}
}
</style>
</head>
<body>
<aside class="stage">
  <div class="brand">
    <i><svg viewBox="0 0 24 24" fill="none" stroke="#0B3A42" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2.5-6 4 12 2.5-6h5"/></svg></i>
    Cover cost estimator
  </div>
  <div>
    <h1>See what your health cover could cost</h1>
    <p class="lede" style="margin-top:12px">Enter a few personal details and get a yearly estimate from a trained XGBoost model.</p>
  </div>

  <svg class="pulse" viewBox="0 0 600 120" preserveAspectRatio="none" aria-hidden="true">
    <path class="track" pathLength="1000" fill="none" stroke-width="2.5" d="M0 60H120L140 60 155 20 175 100 195 40 210 60H330L350 60 365 25 385 95 405 45 420 60H600"/>
    <path class="beat" pathLength="1000" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" d="M0 60H120L140 60 155 20 175 100 195 40 210 60H330L350 60 365 25 385 95 405 45 420 60H600"/>
  </svg>

  <section class="result" aria-live="polite">
    <div class="ring">
      <svg viewBox="0 0 160 160"><circle class="bg" cx="80" cy="80" r="70"/><circle class="fg" id="arc" cx="80" cy="80" r="70"/></svg>
      <b id="ringText">Waiting for details</b>
    </div>
    <div class="figure">
      <small>Estimated yearly charges</small>
      <output id="amount">$0</output>
      <div class="sub" id="monthly"></div>
    </div>
  </section>

  <section class="weights" id="weights" hidden>
    <h2>What the model weighs most</h2>
    <div id="bars"></div>
  </section>
</aside>

<main>
<form id="f" novalidate>
  <h2>Your details</h2>
  <p>All fields are required. Nothing you enter is stored.</p>

  <div class="field">
    <label for="age">Age</label>
    <div class="row">
      <input type="range" id="ageR" min="18" max="100" value="35" aria-label="Age slider">
      <input type="number" id="age" min="18" max="100" value="35" inputmode="numeric">
    </div>
  </div>

  <fieldset class="field">
    <legend>Sex</legend>
    <div class="seg">
      <label><input type="radio" name="sex" value="female" checked><span>Female</span></label>
      <label><input type="radio" name="sex" value="male"><span>Male</span></label>
    </div>
  </fieldset>

  <div class="field">
    <label for="bmi">Body mass index (BMI)</label>
    <div class="row">
      <input type="number" id="bmi" min="10" max="60" step="0.1" value="26.5" inputmode="decimal">
      <span class="chip" id="chip">Overweight</span>
    </div>
    <details>
      <summary>Calculate from height and weight</summary>
      <div class="row">
        <input type="number" id="h" min="100" max="230" placeholder="Height, cm" aria-label="Height in centimetres">
        <input type="number" id="w" min="25" max="250" placeholder="Weight, kg" aria-label="Weight in kilograms">
        <button type="button" id="calc">Use this BMI</button>
      </div>
    </details>
  </div>

  <div class="field">
    <label id="chL">Children covered</label>
    <div class="step" role="group" aria-labelledby="chL">
      <button type="button" id="minus" aria-label="Fewer children">&minus;</button>
      <output id="kids">0</output>
      <button type="button" id="plus" aria-label="More children">+</button>
    </div>
  </div>

  <fieldset class="field">
    <legend>Do you smoke?</legend>
    <div class="seg smoke">
      <label><input type="radio" name="smoker" value="no" checked><span>No</span></label>
      <label><input type="radio" name="smoker" value="yes"><span>Yes</span></label>
    </div>
  </fieldset>

  <fieldset class="field">
    <legend>Region</legend>
    <div class="seg four">
      <label><input type="radio" name="region" value="northeast" checked><span>Northeast</span></label>
      <label><input type="radio" name="region" value="northwest"><span>Northwest</span></label>
      <label><input type="radio" name="region" value="southeast"><span>Southeast</span></label>
      <label><input type="radio" name="region" value="southwest"><span>Southwest</span></label>
    </div>
  </fieldset>

  <button class="go" id="go" type="submit"><span class="spin"></span><span id="goT">Estimate cost</span></button>
  <div class="err" id="err" role="alert"></div>
  <p class="fine">Figures are in US dollars. This is a statistical estimate, not an insurance quote.</p>
</form>
</main>

<script>
const IMPORTANCE = __IMPORTANCE__;
const $ = s => document.querySelector(s);
const age = $('#age'), ageR = $('#ageR'), bmi = $('#bmi'), chip = $('#chip');
let kids = 0;

/* age slider <-> number */
ageR.addEventListener('input', () => age.value = ageR.value);
age.addEventListener('input', () => { if (age.value) ageR.value = Math.min(100, Math.max(18, age.value)); });

/* BMI category chip */
function bmiChip() {
  const v = parseFloat(bmi.value);
  let t = 'Enter BMI', c = 'chip';
  if (v > 0) {
    if (v < 18.5) { t = 'Underweight'; c = 'chip low'; }
    else if (v < 25) { t = 'Healthy range'; }
    else if (v < 30) { t = 'Overweight'; c = 'chip warn'; }
    else { t = 'Obesity range'; c = 'chip high'; }
  }
  chip.textContent = t; chip.className = c;
}
bmi.addEventListener('input', bmiChip); bmiChip();

$('#calc').addEventListener('click', () => {
  const h = parseFloat($('#h').value) / 100, w = parseFloat($('#w').value);
  if (h > 0 && w > 0) { bmi.value = (w / (h * h)).toFixed(1); bmiChip(); }
});

/* children stepper */
const setKids = n => { kids = Math.min(10, Math.max(0, n)); $('#kids').textContent = kids; };
$('#minus').onclick = () => setKids(kids - 1);
$('#plus').onclick = () => setKids(kids + 1);

/* model weights */
if (IMPORTANCE.length) {
  const max = Math.max(...IMPORTANCE.map(d => d.value));
  $('#bars').innerHTML = IMPORTANCE.map(d =>
    `<div class="w"><span>${d.name}</span><div><em data-w="${(d.value / max * 100).toFixed(1)}"></em></div><span>${d.value}%</span></div>`).join('');
  $('#weights').hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() =>
    document.querySelectorAll('.w em').forEach(e => e.style.width = e.dataset.w + '%')));
}

/* result animation */
const usd = new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', maximumFractionDigits: 0});
const CIRC = 440, SCALE = 60000;
let shown = 0, raf;
function countTo(target) {
  cancelAnimationFrame(raf);
  const from = shown, t0 = performance.now(), dur = matchMedia('(prefers-reduced-motion:reduce)').matches ? 0 : 1000;
  const tick = now => {
    const p = dur ? Math.min(1, (now - t0) / dur) : 1, e = 1 - Math.pow(1 - p, 3);
    shown = from + (target - from) * e;
    $('#amount').textContent = usd.format(shown);
    if (p < 1) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

/* submit */
$('#f').addEventListener('submit', async ev => {
  ev.preventDefault();
  const err = $('#err'), go = $('#go'); err.textContent = '';
  const body = {
    age: age.value, bmi: bmi.value, children: kids,
    sex: document.querySelector('[name=sex]:checked').value,
    smoker: document.querySelector('[name=smoker]:checked').value,
    region: document.querySelector('[name=region]:checked').value
  };
  go.disabled = true; $('#goT').textContent = 'Estimating';
  try {
    const r = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Something went wrong.');
    countTo(d.charges);
    $('#monthly').textContent = 'About ' + usd.format(d.monthly) + ' a month';
    $('#ringText').textContent = 'Scale: $0 to $60,000';
    $('#arc').style.strokeDashoffset = CIRC * (1 - Math.min(d.charges / SCALE, 1));
    if (matchMedia('(max-width:900px)').matches) $('.stage').scrollIntoView({behavior: 'smooth'});
  } catch (e) {
    err.textContent = e.message;
  } finally {
    go.disabled = false; $('#goT').textContent = 'Estimate again';
  }
});
</script>
</body>
</html>
"""

PAGE = PAGE.replace("__IMPORTANCE__", json.dumps(IMPORTANCE))


@app.get("/")
def index():
    return PAGE


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
