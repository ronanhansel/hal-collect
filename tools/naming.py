"""
Shared naming and normalization utilities for test taker identification.
Used across multiple scripts to ensure consistent naming conventions.
"""

import re
import pandas as pd


def normalize_date(text):
    """
    Decodes dates into YYYYMMDD format.
    Handles YYYY-MM-DD, YYYY_MM_DD, YYYY/MM/DD.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Pattern to capture YYYY, MM, DD separated by -, _, /, or nothing
    def replace_date(match):
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    
    pattern = re.compile(r'(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})')
    return pattern.sub(replace_date, text)


def extract_reasoning(text):
    """
    Extracts reasoning effort keywords from a string.
    Keywords: high, medium, low, minimal, thinking, no reasoning.
    """
    if not isinstance(text, str):
        return []
    keywords = ['high', 'medium', 'low', 'minimal', 'thinking', 'no reasoning']
    found = []
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            if kw == 'no reasoning':
                found.append('no_reasoning')
            else:
                found.append(kw)
    return found


def clean_string(text):
    """
    Standardizes string: 
    - Converts to LOWERCASE.
    - Replaces all punctuation and spaces with '_'
    - Removes duplicate underscores
    - Strips leading/trailing underscores
    """
    if not isinstance(text, str):
        return str(text)
    
    # convert to lower case
    text = text.lower()
    
    # Replace non-alphanumeric characters with _
    text = re.sub(r'[^a-z0-9]', '_', text)
    # Remove multiple underscores
    text = re.sub(r'_+', '_', text)
    # Strip underscores
    return text.strip('_')


def fix_gpt_corruption(text):
    """
    Fixes the specific data corruption where 'o' appears to have been replaced by 'gpt-o'.
    """
    return text.replace('gpt_o', 'o')


def resolve_model_alias(model_name):
    """
    Maps short or inconsistent model names to their canonical, full-date versions
    to prevent duplication (e.g., gpt_5 vs gpt_5_20250807).
    """
    mapping = {
        'claude_3_7_sonnet': 'claude_3_7_sonnet_20250219',
        'claude_sonnet_4_5': 'claude_sonnet_4_5_20250929',
        'claude_sonnet_4': 'claude_sonnet_4_20250514',
        'gpt_5': 'gpt_5_20250807',
        'gpt_4_1': 'gpt_4_1_20250414',
        'gemini_2_0_flash_001': 'gemini_2_0_flash',
        'gemini_gemini_2_0_flash': 'gemini_2_0_flash' # Fix duplicate word
    }
    
    # Handle the specific together_ai prefix
    if 'together_ai_deepseek_ai_' in model_name:
        model_name = model_name.replace('together_ai_deepseek_ai_', '')
        
    return mapping.get(model_name, model_name)


def clean_model_name_logic(text, reasoning_list):
    """
    Cleans the model name by applying all normalization and fixes.
    """
    if not isinstance(text, str):
        return ""
    
    text_clean = text.lower() # Ensure input is lower
    
    # Remove reasoning keywords
    for kw in reasoning_list:
        kw_pattern = kw.replace('_', '[ _]')
        text_clean = re.sub(f"(?i){re.escape(kw_pattern)}", "", text_clean)

    # Normalize dates
    text_clean = normalize_date(text_clean)
    
    # Standardize string (lower, punct -> _)
    text_clean = clean_string(text_clean)
    
    # Fix corruptions and aliases
    text_clean = fix_gpt_corruption(text_clean)
    text_clean = resolve_model_alias(text_clean)
    
    return text_clean


def get_scaffold(agent_name):
    """
    Identifies the scaffold name. Returns LOWERCASE standardized strings.
    """
    an_lower = str(agent_name).lower()
    
    # Priority Regex Mapping
    patterns = [
        (r'assistant.*bench.*browser.*agent', 'assistantbench_browser_agent'),
        (r'hal.*generalist', 'hal_generalist_agent'),
        (r'browser.*use.*test', 'browser_use_test'),
        (r'browser.*use', 'browser_use'),
        (r'colbench.*example.*agent', 'colbench_example_agent'),
        (r'colbench.*text', 'colbench_text'),
        (r'core.*agent', 'core_agent'),
        (r'hf.*open.*deep.*research', 'hf_open_deep_research'),
        (r'scicode.*tool.*calling.*agent', 'scicode_tool_calling_agent'),
        (r'scicode.*zero.*shot.*agent', 'scicode_zero_shot_agent'),
        (r'sab.*self.*debug', 'sab_self_debug'),
        (r'taubench.*tool.*calling', 'taubench_toolcalling'),
        (r'tau.*bench.*few.*shot', 'taubench_fewshot'), # Catches "TAU-bench FewShot"
        (r'usaco.*episodic.*semantic', 'usaco_episodic_semantic'),
        (r'swe.*agent', 'swe_agent'),
        (r'my.*agent', 'my_agent'),
        (r'seeact', 'seeact')
    ]
    
    for pat, name in patterns:
        if re.search(pat, an_lower):
            return name
            
    # Fallback
    if '(' in str(agent_name):
        s = str(agent_name).split('(')[0]
    else:
        s = str(agent_name)
        
    return clean_string(s)


def generate_test_taker_id(agent_name, model_name):
    """
    Generates the unique test_taker_id using format: scaffold:model_effort
    
    Args:
        agent_name: Agent name from data
        model_name: Model name from data (can be NaN)
        
    Returns:
        String in format: scaffold:model_effort
    """
    # 1. Extract Scaffold
    scaffold = get_scaffold(agent_name)
    
    # 2. Determine Raw Model Name
    raw_model = ""
    if pd.notna(model_name) and str(model_name).lower() != 'nan':
        raw_model = str(model_name)
    else:
        match = re.search(r'\((.*?)\)', str(agent_name))
        if match:
            raw_model = match.group(1)
        else:
            raw_model = "unknown"
            
    # 3. Extract Reasoning
    r1 = extract_reasoning(str(agent_name))
    r2 = extract_reasoning(raw_model)
    reasoning_set = set(r1 + r2)
    reasoning_list = sorted(list(reasoning_set))
    reasoning_str = "_".join(reasoning_list)
    
    # 4. Clean and Fix Model Name
    clean_model = clean_model_name_logic(raw_model, reasoning_list)
    
    # 5. Construct ID: scaffold:model_effort
    if reasoning_str:
        return f"{scaffold}:{clean_model}_{reasoning_str}"
    else:
        return f"{scaffold}:{clean_model}"


# Legacy compatibility functions for match_rubrics.py

def normalize_model_name(model_str):
    """
    Legacy function name for backward compatibility.
    Normalizes a model name string.
    """
    if not isinstance(model_str, str):
        return str(model_str)
    
    # Extract reasoning keywords
    reasoning = extract_reasoning(model_str)
    
    # Clean the model name
    clean_model = clean_model_name_logic(model_str, reasoning)
    
    # If reasoning was found, append it
    if reasoning:
        reasoning_str = "_".join(sorted(reasoning))
        return f"{clean_model}_{reasoning_str}"
    
    return clean_model


def normalize_scaffold_name(scaffold_str):
    """
    Legacy function name for backward compatibility.
    Normalizes a scaffold name string.
    """
    return clean_string(scaffold_str)
