# Hermes Stats

Statistieken dashboard voor [Hermes](https://github.com/NousResearch/hermes-agent) sessies en systeemgebruik. Gebouwd met FastAPI en Chart.js.

## Functionaliteit

**Overzichtsscherm**
- GPU status met temperatuur, gebruik, VRAM-verbruik en geladen Ollama modellen (live, elke 10 seconden)
- Statistiekkaarten: sessies, turns, tokens, duur, reactietijd, tool calls, compressie, API-type
- Grafieken: sessies per dag (30 dagen) en top gebruikte tools
- Sessietabel met alle metrics per sessie — klik om in te zoomen

**Sessiedetail**
- Turn-by-turn uitsplitsing: duur, tokens, gebruikte tools
- Context window visualisatie (gebruik vs. maximum)
- Compressiestatus

**Metrics**

| Metric | Bron |
|---|---|
| Tokens, turns, tool calls | `state.db` (exact) of JSONL/messages (geschat, gemarkeerd met `~`) |
| Reactietijd per turn | `agent.log` |
| GPU: temp, gebruik, VRAM | `nvidia-smi` (live) |
| Geladen modellen | `ollama ps` (live) |
| Context window % | input tokens / `ollama_num_ctx` (40 960) |
| Compressie | inactieve berichten in `state.db` |
| API lokaal vs. cloud | `billing_base_url` in `state.db` |

## Installatie

Vereist de Hermes virtual environment (`~/.hermes/hermes-agent/venv/`) — FastAPI en uvicorn zijn daarin al aanwezig.

```bash
git clone http://ugreendxp2800.local:3000/Paikke/hermes-statistieken.git ~/.hermes/dashboard
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
| `AGENT_LOG` | `~/.hermes/logs/agent.log` | Logbestand voor reactietijden |
| `MAX_CTX` | `40960` | Context window maximum (tokens) |
| Poort | `8088` | Luisterpoort uvicorn |

## Stack

- **Backend:** Python 3.11, FastAPI, uvicorn
- **Frontend:** Vanilla JS, Chart.js 4.4
- **Data:** SQLite (`state.db`), JSONL sessiebestanden, `agent.log`
