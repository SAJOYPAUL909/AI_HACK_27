import json
import requests
from config import Config

class LLMClient:
    def __init__(self):
        self.api_key = Config.OPENAI_API_KEY
        self.api_url = Config.OPENAI_API_URL

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        If OPENAI_API_KEY is set, perform an external call (OpenAI-compatible).
        Otherwise, use a simple rule-based generator to produce JSON recommendations.
        """
        if self.api_key and self.api_key.strip() != "":
            return self._call_external(system_prompt, user_prompt)
        return self._rule_based(system_prompt, user_prompt)

    def _call_external(self, system_prompt, user_prompt):
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 600,
            "temperature": 0.2
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        r = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    def _rule_based(self, system_prompt, user_prompt):
        """
        Very small heuristic generator that extracts a few metrics and returns JSON suggestions.
        The prompt is expected to contain strings like: total_kwh=XXX, peak_hour=HH, price_per_kwh=Y.YY
        """
        import re
        def find_num(k):
            m = re.search(rf"{k}=([0-9\\.]+)", user_prompt)
            return float(m.group(1)) if m else None

        total_kwh = find_num("total_kwh")
        avg = find_num("avg_hourly_kwh")
        peak = find_num("peak_hour")
        price = find_num("price_per_kwh") or 0.15

        immediate = []
        scheduled = []
        investment = []

        if total_kwh and total_kwh > 100:
            immediate.append({
                "title": "Check HVAC setpoints and schedules",
                "description": "Adjust thermostat and schedules to reduce excessive heating/cooling.",
                "estimated_kwh_savings": round(0.05 * total_kwh, 2),
                "estimated_usd_savings": round(0.05 * total_kwh * price, 2)
            })

        immediate.append({
            "title": "Unplug phantom loads / use smart strips",
            "description": "Reduces standby consumption from peripherals and chargers.",
            "estimated_kwh_savings": round(0.01 * (total_kwh or 100), 2),
            "estimated_usd_savings": round(0.01 * (total_kwh or 100) * price, 2)
        })

        scheduled.append({
            "title": "Shift flexible loads to off-peak hours",
            "description": "Schedule EV charging, dishwashers, and laundry during off-peak/night.",
            "estimated_kwh_savings": 0,
            "estimated_usd_savings": 0
        })

        investment.append({
            "title": "Install programmable/smart thermostat and sensors",
            "description": "Improves setpoint control and can save energy long-term.",
            "estimated_kwh_savings": round(0.1 * (total_kwh or 100), 2),
            "estimated_usd_savings": round(0.1 * (total_kwh or 100) * price, 2)
        })

        return json.dumps({
            "immediate": immediate,
            "scheduled": scheduled,
            "investment": investment
        })
