import os
import json
from glob import glob

def load_json_files(directory):
    """Loads all JSON files in a directory and returns a dictionary with method names as keys."""
    method_mapping = {
        "h2o_true": "Prefill Eviction Method",
        "oracle": "Oracle",
        "quest": "Page Based Method",
        "ExpPred": "TokenButler"
    }
    
    data = {}
    for method, title in method_mapping.items():
        # file_path = os.path.join(directory, f"{method}.json")
        file_path = os.path.join(directory, method, "latex_text.json")

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data[title] = json.load(f)
    return data

# def generate_latex(data, output_dir):
#     """Generates LaTeX files with formatted tcolorboxes from loaded JSON data."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Assume all methods have the same number of examples
#     num_examples = len(next(iter(data.values())))
#     num_files = min(10, num_examples)  # Generate up to 10 files
    
#     for i in range(num_files):
#         latex_content = r"""
#         \documentclass{article}
#         \usepackage[most]{tcolorbox}
#         \usepackage[a4paper,margin=1in]{geometry}
#         \begin{document}
#         """
        
#         for method, title in data.items():
#             example_list = list(title.values())  # Ensure it's a list
#             example_text = example_list[i] if i < len(example_list) else ""
#             # example_text = title[i] if i < len(title) else ""
#             latex_content += fr"""
#             \begin{{tcolorbox}}[
#                 colframe=black!60,
#                 colback=blue!5,
#                 coltitle=white,
#                 sharp corners,
#                 boxrule=0.8pt,
#                 left=4pt, right=4pt, top=6pt, bottom=6pt,
#                 fontupper=\ttfamily,
#                 before skip=8pt,
#                 after skip=8pt,
#                 width=\columnwidth,
#                 title={{\centering {method}}}
#             ]
#             {example_text}
#             \end{{tcolorbox}}
#             """
        
#         latex_content += "\\end{document}"
        
#         output_file = os.path.join(output_dir, f"latex_{i}.tex")
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(latex_content)
def generate_latex(data, output_dir):
    """Generates LaTeX files with formatted tcolorboxes arranged in a 2x2 grid."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_examples = len(next(iter(data.values())))
    num_files = min(10, num_examples)  # Generate up to 10 files
    
    for i in range(num_files):
        latex_content = r"""
        \documentclass{article}
        \usepackage[most]{tcolorbox}
        \usepackage[a4paper,margin=1in]{geometry}
        \usepackage{multicol}
        \begin{document}
        \noindent
        \begin{minipage}{0.48\linewidth}
        """
        
        count = 0  # Track the number of items in the current file
        for method, title in data.items():
            example_list = list(title.values())
            example_text = example_list[i] if i < len(example_list) else ""

            latex_content += fr"""
            \begin{{tcolorbox}}[
                colframe=black!60,
                colback=blue!5,
                coltitle=white,
                sharp corners,
                boxrule=0.8pt,
                left=4pt, right=4pt, top=6pt, bottom=6pt,
                fontupper=\ttfamily,
                before skip=8pt,
                after skip=8pt,
                width=\textwidth,
                title={{\centering {method}}}
            ]
            \fontsize{{6}}{{7}}\selectfont {example_text} % Explicitly setting small font
            \end{{tcolorbox}}
            """

            count += 1
            if count % 2 == 0:  # After 2 entries, close and open new minipage
                latex_content += r"""
                \end{minipage}
                \hfill
                \begin{minipage}{0.48\linewidth}
                """

        latex_content += r"""
        \end{minipage}
        \end{document}
        """
        
        output_file = os.path.join(output_dir, f"latex_{i}.tex")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_content)
def main():
    base_dir = "./latex_traces/"
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(model_dirs)
    if not model_dirs:
        print("No model directories found.")
        return
    
    model_dir = os.path.join(base_dir, model_dirs[0])  # Pick the first model directory
    eval_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    
    if not eval_dirs:
        print("No eval directories found.")
        return
    
    data = load_json_files(model_dir)
    output_dir = os.path.join(base_dir, "generated_latex")
    generate_latex(data, output_dir)
    print(f"Generated LaTeX files in {output_dir}")

if __name__ == "__main__":
    main()
