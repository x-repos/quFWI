import re

with open('manuscript.tex', 'r') as f:
    text = f.read()

# Extract Table 1
table_match = re.search(r'(\\begin\{table\}.*?\\end\{table\})', text, flags=re.DOTALL)
if table_match:
    table_text = table_match.group(1)
    text = text.replace(table_text, '')
    
    # Extract Figures 1 and 2
    figures = re.findall(r'(\\begin\{figure\}.*?\\end\{figure\})', text, flags=re.DOTALL)
    
    if len(figures) >= 2:
        fig1_text = figures[0]
        fig2_text = figures[1]
        
        text = text.replace(fig1_text, '')
        text = text.replace(fig2_text, '')
        
        insert_target = "Full methodological details, including the source-free acoustic wave equation, the FBPINN loss construction, and the PQC implementation, are given in Methods."
        
        if insert_target in text:
            new_insert = f"{insert_target}\n\n{fig1_text}\n\n{fig2_text}\n\n{table_text}"
            text = text.replace(insert_target, new_insert)
            
            with open('manuscript.tex', 'w') as f:
                f.write(text)
            print("Successfully moved Figure 1, Figure 2, and Table 1.")
        else:
            print("Could not find insert target.")
else:
    print("Could not find table.")
