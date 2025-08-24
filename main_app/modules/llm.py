from typing import Optional, Any
import os
import json
import base64
import io

from dotenv import load_dotenv  # type: ignore
from PIL import Image  # type: ignore


class LLMInterpreter:
    def __init__(self) -> None:
        load_dotenv()
        # Groq SDK reads GROQ_API_KEY from env
        try:
            from groq import Groq  # type: ignore
        except Exception as exc:  # pragma: no cover
            self.client = None  # type: ignore[assignment]
            self.error: Optional[str] = f"groq package not installed: {exc}"
            return
        self.client = Groq()
        self.error = None

    def analyze(self, image_path: str, save_path: Optional[str] = None, retries: int = 3) -> Optional[str]:
        if self.client is None:
            return None

        # Encode the image as data URL (do not change the user's prompt text)
        with Image.open(image_path).convert('RGB') as im:
            buf = io.BytesIO()
            im.save(buf, format='JPEG')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{b64}"

        # Type ignores to accommodate SDK types
        messages_obj: Any = [  # type: ignore[var-annotated]
              {
                "role": "system",
                "content": """# Historical Photo Analysis Prompt

You are an expert historical photo analyst with advanced vision capabilities. Your task is to analyze historical photographs and provide structured information about their content **based on what there is on the photo and not assumptions.**

## Instructions

For each historical photograph provided, analyze the image carefully and provide the following information in the exact JSON format specified below:

### Analysis Requirements:

1. **Caption**: Create a concise 6-8 word caption that captures the essence of the photo
2. **Minors Count**: Count the number of people who appear to be under 18 years old
3. **Jewish Symbols**: Identify if any Jewish religious or cultural symbols are present (Star of David, menorah, Torah scrolls, Hebrew text, religious clothing like yarmulkes/kippahs, tallitot, etc.)
4. **Nazi Symbols**: Identify if any Nazi regime symbols are present (swastikas, Nazi eagles, SS symbols, Nazi uniforms, Nazi flags, etc.)
5. **Text Analysis**: Identify any Hebrew or German text visible in the image and provide translation/description
6. **Objects, Artifacts & Animals**: Identify up to 5 main objects, historical artifacts, or animals visible in the photo
7. **Violence Assessment**: Determine if there are any visible signs of violence, conflict, or distress

### Critical Symbol Identification Guidelines:

**IMPORTANT - Nazi Symbol Detection:**
- Only identify symbols as "Nazi" if there are CLEAR, SPECIFIC Nazi regime indicators present
- Do NOT assume Nazi presence based solely on: German soldiers, German uniforms, wartime context, or WWII timeframe
- REQUIRE specific Nazi insignia such as: swastikas, SS runes, Nazi party eagles, Hitler portraits, Nazi flags
- If you see German military without specific Nazi symbols, categorize as "German army" or "wartime"
- Distinguish between general wartime/military imagery and specific Nazi regime symbols
- Historical context alone is NOT sufficient - you must see actual Nazi symbols or insignia

### Important Guidelines:

- Be thorough but objective in your analysis
- Count only clearly visible people for age assessment
- Look carefully for symbols that may be partially visible or in the background
- Consider historical context when identifying symbols
- If uncertain about a person's age, err on the side of caution
- Distinguish between Nazi symbols and other similar-looking historical symbols
- Identify prominent objects, artifacts, and animals that help tell the story of the photo
- Look for signs of violence, weapons, distress, destruction, or conflict
- Consider historical context and wartime imagery
- **BE PRECISE**: Only mark as Nazi symbols when actual Nazi insignia are visible, not just German military or WWII context

## Required Output Format

Provide your analysis in this exact JSON structure:

```json
{
  "caption": "[6-8 word description of the photo]",
  "people_under_18": [number],
  "has_jewish_symbols": [true/false],
  "jewish_symbols_details": [
    {
      "symbol_type": "[type of symbol]",
      "description": "[detailed description of what it is]",
      "location_in_image": "[where it appears in the photo]"
    }
  ],
  "has_nazi_symbols": [true/false],
  "nazi_symbols_details": [
    {
      "symbol_type": "[type of symbol]",
      "description": "[detailed description of what it is]",
      "location_in_image": "[where it appears in the photo]"
    }
  ],
  "text_analysis": {
    "hebrew_text": {
      "present": [true/false],
      "text_found": "[actual Hebrew text if visible]",
      "translation": "[English translation]",
      "context": "[what type of text - religious, signage, etc.]"
    },
    "german_text": {
      "present": [true/false],
      "text_found": "[actual German text if visible]",
      "translation": "[English translation]",
      "context": "[what type of text - signage, documents, etc.]"
    }
  },
  "main_objects_artifacts_animals": [
    {
      "item": "[name of object/artifact/animal]",
      "category": "[object/artifact/animal]",
      "description": "[brief description of the item]",
      "significance": "[historical or contextual importance]"
    }
  ],
  "violence_assessment": {
    "signs_of_violence": [true/false],
    "explanation": "[detailed explanation of what violence indicators are visible, or 'No signs of violence detected' if false]"
  }
}
```

## Example Outputs:

### Example 1: Nazi Rally Photo
```json
{
  "caption": "Nazi party rally with swastika flags",
  "people_under_18": 0,
  "has_jewish_symbols": false,
  "jewish_symbols_details": [],
  "has_nazi_symbols": true,
  "nazi_symbols_details": [
    {
      "symbol_type": "Swastika flag",
      "description": "Large red Nazi flag with black swastika in white circle",
      "location_in_image": "Multiple flags hanging from building facade"
    },
    {
      "symbol_type": "Nazi eagle",
      "description": "Nazi party eagle with spread wings clutching swastika",
      "location_in_image": "Above podium on building"
    }
  ],
  "text_analysis": {
    "hebrew_text": {
      "present": false,
      "text_found": "",
      "translation": "",
      "context": ""
    },
    "german_text": {
      "present": true,
      "text_found": "Deutschland über alles",
      "translation": "Germany above all",
      "context": "Banner text visible on building"
    }
  },
  "main_objects_artifacts_animals": [
    {
      "item": "Nazi flags",
      "category": "artifact",
      "description": "Red flags with black swastikas",
      "significance": "Nazi party propaganda and territorial marking"
    },
    {
      "item": "Podium",
      "category": "object",
      "description": "Raised speaking platform with Nazi insignia",
      "significance": "Platform for Nazi political speeches"
    },
    {
      "item": "Crowd barriers",
      "category": "object",
      "description": "Wooden barriers controlling crowd movement",
      "significance": "Crowd control for mass political gathering"
    }
  ],
  "violence_assessment": {
    "signs_of_violence": false,
    "explanation": "No direct violence visible, though this represents a gathering of a violent political regime"
  }
}
```

### Example 2: World War I Soldiers
```json
{
  "caption": "German soldiers in WWI trench warfare",
  "people_under_18": 1,
  "has_jewish_symbols": false,
  "jewish_symbols_details": [],
  "has_nazi_symbols": false,
  "nazi_symbols_details": [],
  "text_analysis": {
    "hebrew_text": {
      "present": false,
      "text_found": "",
      "translation": "",
      "context": ""
    },
    "german_text": {
      "present": true,
      "text_found": "Gott mit uns",
      "translation": "God with us",
      "context": "Text on German military belt buckle"
    }
  },
  "main_objects_artifacts_animals": [
    {
      "item": "Rifles",
      "category": "object",
      "description": "WWI-era military rifles with bayonets",
      "significance": "Standard infantry weapons of the period"
    },
    {
      "item": "Steel helmets",
      "category": "object",
      "description": "German Stahlhelm protective headgear",
      "significance": "Iconic WWI German military equipment"
    },
    {
      "item": "Trench fortifications",
      "category": "object",
      "description": "Wooden and earthwork defensive structures",
      "significance": "Characteristic WWI defensive warfare infrastructure"
    },
    {
      "item": "Military backpack",
      "category": "object",
      "description": "Canvas field pack with personal supplies",
      "significance": "Essential equipment for extended battlefield deployment"
    }
  ],
  "violence_assessment": {
    "signs_of_violence": true,
    "explanation": "Military combat setting with visible weapons (rifles with bayonets) and fortified positions indicating active warfare context"
  }
}
```

Now, please analyze the historical photograph provided and return your response in the specified JSON format."""
              },
              {  # type: ignore[dict-item]
                "role": "user",
                "content": [  # type: ignore[list-item]
                  {"type": "image_url", "image_url": {"url": data_url}}  # type: ignore[dict-item]
                ]
              }
            ]
        
        # Retry wrapper for API call
        completion: Any = None
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            try:
                completion = self.client.chat.completions.create(  # type: ignore[call-arg]
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=messages_obj,
                    temperature=0.2,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                    response_format={"type": "json_object"},
                    stop=None
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                # backoff
                try:
                    import time as _t
                    _t.sleep(3.0 * (attempt + 1))
                except Exception:
                    pass
        if completion is None:
            err_obj = {"error": f"LLM API rejected or unreachable: {str(last_exc) or 'unknown'}", "error_type": "api_error"}
            if save_path:
                try:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(err_obj, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            print(json.dumps(err_obj, ensure_ascii=False))
            return json.dumps(err_obj, ensure_ascii=False)

        # Non-streaming JSON response
        full = completion.choices[0].message.content or ""  # type: ignore[attr-defined]
        # Try to extract JSON only and print
        start = full.find('{')
        end = full.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = full[start:end+1]
            print(json_str)
            if save_path:
                try:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(json_str)
                except Exception:
                    pass
            return json_str
        else:
            # Treat as bad JSON
            err_obj = {"error": "LLM returned non-JSON or malformed JSON", "error_type": "bad_json", "raw": full[:5000]}
            print(json.dumps(err_obj, ensure_ascii=False))
            if save_path:
                try:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(err_obj, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            return json.dumps(err_obj, ensure_ascii=False)


