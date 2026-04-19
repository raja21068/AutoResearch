#!/usr/bin/env bash
set -e
echo "╔═══════════════════════════════════════════╗"
echo "║  🧠 NexusAI — One-Click Deploy            ║"
echo "║  AutoCodeAI + ResearchClaw combined        ║"
echo "╚═══════════════════════════════════════════╝"
if [ ! -f .env ]; then cp .env.example .env
  read -p "Enter OpenAI API key (or Enter to skip): " K
  [ -n "$K" ] && sed -i "s|sk-YOUR-KEY-HERE|$K|" .env && echo "✓ Key saved"
fi
mkdir -p workspace output data
[ ! -d .venv ] && python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -q 2>&1 | tail -2
echo ""
echo "✓ Ready! Run:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo "  → http://localhost:8000"
