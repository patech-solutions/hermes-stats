# Hermes Stats — Token Gebruik & Systeem Dashboard

Statistieken dashboard voor [Hermes](https://github.com/NousResearch/hermes-agent) sessies, **Mistral API token gebruik** en systeemgebruik. Gebouwd met FastAPI en Chart.js.

## Functionaliteit

### 🎯 Overzichtsscherm
- **GPU status** met temperatuur, gebruik, VRAM-verbruik en geladen Ollama modellen (live, elke 10 seconden)
- **Mistral API Token Gebruik** — real-time kosten tracking per model (refresh elke 30 seconden)
  - Totaal kosten (€)
  - API calls per model
  - Tokens per model
  - Kostenverdeling visualisatie
- Statistiekkaarten: sessies, turns, tokens, duur, reactietijd, tool calls, compressie, API-type
- **Nieuwe Grafieken:**
  - Token kosten per model (30 dagen)
  - Dagelijkse token kosten trend
  - Sessies per dag (30 dagen)
  - Top gebruikte tools
- Sessietabel met alle metrics per sessie — klik om in te zoomen

### 📊 Sessiedetail
- Turn-by-turn uitsplitsing: duur, tokens, gebruikte tools
- Context window visualisatie (gebruik vs. maximum)
- Compressiestatus

### 💰 Mistral Token Tracking (NIEUW!)
Real-time tracking van Mistral API token gebruik en kosten:

**API Endpoints:**
- `/api/token-usage` — Totale token usage en kosten (per model)
- `/api/models` — Per-model statistieken
- `/api/token-usage/daily` — Dagelijkse token usage
- `/api/token-usage/sessions` — Token usage per sessie

**Bronnen:**
- `agent.log` — API calls met input/output/cache tokens
- Prijzen gebaseerd op Mistral API pricing (juni 2026)

**Ondersteunde Modellen:**
| Model | Input (€/1K) | Output (€/1K) | Cache (€/1K) |
|-------|--------------|---------------|--------------|
| mistral-large-2512 | €0.002 | €0.006 | €0.001 |
| codestral-latest | €0.0005 | €0.0015 | €0.00025 |
| mistral-small-2603 | €0.00025 | €0.00075 | €0.000125 |
| mistral-tiny-latest | €0.00008 | €0.00024 | €0.00004 |
| mistral-medium-3-5 | €0.0007 | €0.0021 | €0.00035 |

**Metrics**

| Metric | Bron |
|---|---|
| Tokens, turns, tool calls | `state.db` (exact) of JSONL/messages (geschat, gemarkeerd met `~`) |
| Reactietijd per turn | `agent.log` |
| **Token kosten** | `agent.log` (API calls met token counts) |
| GPU: temp, gebruik, VRAM | `nvidia-smi` (live) |
| Geladen modellen | `ollama ps` (live) |
| Context window % | input tokens / `MAX_CTX` (64 000) |
| Compressie | inactieve berichten in `state.db` |
| API lokaal vs. cloud | `billing_base_url` in `state.db` |

## Installatie

Vereist de Hermes virtual environment (`~/.hermes/hermes-agent/venv/`) — FastAPI en uvicorn zijn daarin al aanwezig.

```bash
git clone https://github.com/patech-solutions/hermes-stats.git ~/.hermes/dashboard
```

### Systemd service (aanbevolen)

```bash
cp ~/.hermes/dashboard/hermes-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-dashboard.service
```

De service start automatisch mee met de gebruikerssessie.

### Handmatig starten

```bash
bash ~/.hermes/dashboard/start.sh
```

Bereikbaar op `http://localhost:8088`.

## Configuratie

Standaardwaarden in `app.py`:

| Variabele | Waarde | Omschrijving |
|---|---|---|
| `DB_PATH` | `~/.hermes/state.db` | Hermes SQLite database |
| `SESSIONS_DIR` | `~/.hermes/sessions/` | JSONL sessiemap |
| `AGENT_LOG` | `~/.hermes/logs/agent.log` | Logbestand voor reactietijden **en token tracking** |
| `MAX_CTX` | `64000` | Context window maximum (tokens) — **geupdate voor Hermes** |
| Poort | `8088` | Luisterpoort uvicorn |
| `MISTRAL_PRICING` | Zie onder | Prijzen per model voor kostenberekening |

### Mistral Pricing Configuratie
De prijsconfiguratie in `app.py` kan worden aangepast als Mistral hun prijzen wijzigt:

```python
MISTRAL_PRICING = {
    'mistral-large-2512': {'input': 0.002, 'output': 0.006, 'cache': 0.001},
    'codestral-latest': {'input': 0.0005, 'output': 0.0015, 'cache': 0.00025},
    # ... andere modellen
}
```

**Let op:** Cache prijzen zijn geschat. Werkelijke besparingen hangen af van je Mistral contract.

## Gebruik

### Snel starten

```bash
# Clone de repository
 git clone http://ugreendxp2800.local:3000/Paikke/hermes-statistieken.git ~/.hermes/dashboard

# Start het dashboard
bash ~/.hermes/dashboard/start.sh
```

Open in browser: `http://localhost:8088`

### Token Usage Tracken
- De token tracking werkt **automatisch** door `agent.log` te parsen
- Refresht elke **30 seconden** voor near real-time data
- Toont:
  - Totaal kosten (alle modellen)
  - Kosten per model
  - API calls per model
  - Tokens (input/output/cache) per model
  - Dagelijkse kosten trend

### Voorbeeld Output
```json
{
  "total_calls": 76,
  "total_cost": 0.0921,
  "models": {
    "mistral-large-2512": {
      "calls": 28,
      "in_tokens": 721466,
      "out_tokens": 7080,
      "cache_tokens": 0,
      "cost": 0.0887
    },
    "codestral-latest": {
      "calls": 48,
      "in_tokens": 1442821,
      "out_tokens": 3062,
      "cache_tokens": 0,
      "cost": 0.0034
    }
  },
  "last_updated": "2026-06-11 12:34:56"
}
```

## Stack

- **Backend:** Python 3.11, FastAPI, uvicorn
- **Frontend:** Vanilla JS, Chart.js 4.4
- **Data:** SQLite (`state.db`), JSONL sessiebestanden, `agent.log`
