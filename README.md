# Hermes Stats — Token Gebruik & Systeem Dashboard

Statistieken dashboard voor [Hermes Agent](https://github.com/NousResearch/hermes-agent) sessies, API-kosten en systeemgebruik. Werkt met lokale modellen (Ollama) én cloud-API's. Gebouwd met FastAPI en Chart.js; alles in één bestand (`app.py`).

## Functionaliteit

### 🎯 Overzichtsscherm
- **GPU status** — temperatuur, gebruik, vermogen en VRAM via `nvidia-smi`, plus de geladen Ollama-modellen met grootte, CPU/GPU-verdeling en contextlengte (live, elke 10 seconden). Zonder GPU toont het dashboard een nette "cloud-only" melding.
- **API-kosten** — kosten, calls en tokens per model (refresh elke 30 seconden). Alleen modellen met een bekende cloudprijs krijgen kosten; lokale modellen staan op €0.
- **Statistiekkaarten** — sessies, turns, tokens, duur, reactietijd, tool calls, compressie en de verdeling **lokaal / cloud**.
- **Grafieken** — kosten per model (30 dagen), dagelijkse kostentrend, sessies per dag, meest gebruikte tools.
- **Sessietabel** met alle metrics per sessie en een backend-label (lokaal / Cloud API) — klik om in te zoomen.

### 📊 Sessiedetail
- **Gespreksweergave in twee richtingen** — per turn het volledige gebruikersbericht én de antwoorden van de assistent, inclusief tussentijdse berichten en Hermes-systeemnudges (`[System: …]`), met tijdstempels. Lange berichten zijn in-/uitklapbaar (klik op het bericht of "Alles uitklappen"). Alle tekst wordt HTML-escaped weergegeven.
- Per turn: duur, tokens en gebruikte tools.
- Context window visualisatie (gebruik vs. maximum) en compressiestatus.

## Metrics en bronnen

| Metric | Bron |
|---|---|
| Tokens, turns, tool calls | `state.db` (exact) of JSONL/messages (geschat, gemarkeerd met `~`) |
| Gesprek per turn | `messages`-tabel in `state.db` (fallback: JSONL-sessiebestanden) |
| Reactietijd per turn | `agent.log` |
| API-kosten | `agent.log` (API calls met token counts) × `MISTRAL_PRICING` |
| GPU: temp, gebruik, VRAM | `nvidia-smi` (live) |
| Geladen modellen | Ollama HTTP-API `/api/ps` (live) |
| Context window % | input tokens / `MAX_CTX` |
| Compressie | gecomprimeerde berichten in `state.db` |
| Lokaal vs. cloud | `billing_base_url` in `state.db` (zie hieronder) |

**Lokaal of cloud?** Een sessie telt als *lokaal* als het endpoint `localhost`, een loopback- of privé-IP-adres, een `*.local`/`*.lan`-hostnaam of `host.docker.internal` is, of als er geen endpoint is geregistreerd (oudere Hermes-versies). Een publieke host (bijv. Mistral, OpenRouter) telt als *cloud*.

## API-endpoints

| Endpoint | Omschrijving |
|---|---|
| `/api/sessions` | Sessielijst met metrics (`limit`, `offset`) |
| `/api/sessions/{id}` | Sessiedetail inclusief turns en berichten (`turns[].messages`) |
| `/api/stats` | Statistiekenoverzicht (incl. `local_count`) |
| `/api/gpu` | GPU-status en geladen Ollama-modellen |
| `/api/token-usage` | Token-gebruik en kosten per model (`days`) |
| `/api/token-usage/daily` | Dagelijks token-gebruik |
| `/api/token-usage/sessions` | Token-gebruik per sessie |
| `/api/models` | Per-model statistieken |
| `/api/memory-log` | Geheugengebruik uit `agent.log` |

## Installatie

Vereisten: Python 3.11+ met `fastapi` en `uvicorn`. De Hermes virtual environment (`~/.hermes/hermes-agent/venv/`) bevat deze al; gebruik je systeem-Python, installeer ze dan zelf (`pip install fastapi uvicorn`).

```bash
git clone https://github.com/patech-solutions/hermes-stats.git ~/.hermes/dashboard
```

### Handmatig starten

```bash
bash ~/.hermes/dashboard/start.sh
```

Bereikbaar op `http://localhost:8088`.

### Systemd service (aanbevolen)

Controleer eerst in `hermes-dashboard.service`:
- `WorkingDirectory` — de map waar je de repository hebt gekloond (bijv. `%h/.hermes/dashboard`);
- `ExecStart` — een Python met `fastapi` en `uvicorn`, bijv. `%h/.hermes/hermes-agent/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8088 --log-level warning`.

Dezelfde Python-keuze geldt voor `start.sh`.

```bash
cp ~/.hermes/dashboard/hermes-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hermes-dashboard.service
```

De service start automatisch mee met de gebruikerssessie.

> **Let op:** de service luistert op `0.0.0.0` en heeft geen authenticatie. In de gespreksweergave zijn volledige berichten zichtbaar. Bind op `127.0.0.1` of zet het dashboard achter een reverse proxy met authenticatie als het apparaat in een gedeeld netwerk hangt.

## Configuratie

Standaardwaarden bovenin `app.py`:

| Variabele | Standaard | Omschrijving |
|---|---|---|
| `DB_PATH` | `~/.hermes/state.db` | Hermes SQLite-database |
| `SESSIONS_DIR` | `~/.hermes/sessions/` | JSONL-sessiemap (fallback) |
| `AGENT_LOG` | `~/.hermes/logs/agent.log` | Logbestand voor reactietijden en token-tracking |
| `MAX_CTX` | `131072` | Context window maximum (tokens) |
| `COMPRESSION_THRESHOLD` | `52428` | Token-drempel waarop Hermes comprimeert |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama-API voor geladen modellen |
| `MAX_MSG_CHARS` | `20000` | Max. tekens per bericht in de gespreksweergave |
| `MISTRAL_PRICING` | zie hieronder | Prijzen per model voor kostenberekening |
| Poort | `8088` | Luisterpoort uvicorn (in service/`start.sh`) |

### Prijzen

Kosten worden alleen berekend voor modellen in `MISTRAL_PRICING` (EUR per 1K tokens, juni 2026). Modellen die er niet in staan — zoals lokale Ollama-modellen — kosten €0. Gebruik je een ander cloudmodel, voeg dan het tarief toe:

```python
MISTRAL_PRICING = {
    'mistral-medium-3':   {'input': 0.0004,  'output': 0.002,   'cache': 0.0002},
    'mistral-large-2512': {'input': 0.002,   'output': 0.006,   'cache': 0.001},
    'codestral-latest':   {'input': 0.0005,  'output': 0.0015,  'cache': 0.00025},
    # ... andere modellen
}
```

**Let op:** cacheprijzen zijn geschat; werkelijke kosten hangen af van je contract.

### Voorbeeld `/api/token-usage`

```json
{
  "total_calls": 12,
  "total_cost": 0.1072,
  "models": {
    "mistral-large-2512": {
      "calls": 4,
      "in_tokens": 50000,
      "out_tokens": 1200,
      "cache_tokens": 0,
      "cost": 0.1072
    },
    "llama3.1:8b": {
      "calls": 8,
      "in_tokens": 64000,
      "out_tokens": 2100,
      "cache_tokens": 0,
      "cost": 0.0
    }
  },
  "last_updated": "2026-09-25 12:00:00"
}
```

## Wijzigingen

**2026-09**
- Gespreksweergave in twee richtingen in de sessiedetail (voorheen alleen het gebruikersbericht, afgekapt op 200 tekens); berichttekst wordt HTML-escaped.
- Lokale modellen worden weer correct als *lokaal* getoond in plaats van *Cloud API*; robuustere lokaal/cloud-detectie.
- GPU-status werd onterecht als "cloud-only" gemeld terwijl `nvidia-smi` werkte.
- Geladen modellen via de Ollama-API in plaats van het parsen van `ollama ps`-tekst (kolommen schoven).
- Lokale modellen krijgen geen API-kosten meer (vielen terug op een cloudtarief).

**2026-06**
- API-kostentracking per model uit `agent.log`.
- Compatibiliteit met het huidige Hermes `state.db`-schema; betere afhandeling van setups zonder GPU.

## Stack

- **Backend:** Python 3.11+, FastAPI, uvicorn
- **Frontend:** Vanilla JS, Chart.js 4.4
- **Data:** SQLite (`state.db`), JSONL-sessiebestanden, `agent.log`

## Licentie

MIT — zie [LICENSE](LICENSE).
