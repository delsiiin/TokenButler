import json
import random
from collections import Counter
import statistics
import nltk  # pip install nltk


nltk.download('punkt_tab')
def dataset_statistics(num_generations=10000, data_file="mrcr_custom.json"):
    """
    Generates a set of synthetic sentences, then computes simple statistics.

    Args:
        num_generations (int): Number of synthetic sentences to generate.
        data_file (str): JSON file containing the data pools
                         (location introductions, tiebacks, etc.).
    """
    # Load data from JSON
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    location_names = data.get("location_names", [])
    loc_intro_tieback_list = data.get("loc_intro_tieback_list", [])
    philosophy_statements = data.get("philosophy_statements", [])
    food_descriptions = data.get("food_descriptions", [])
    math_problems = data.get("math_problems", [])

    # Generate samples
    all_samples = []
    used_locations = []
    
    random.seed(42)
    for i in range(num_generations):
        locintro_tieback = random.choice(loc_intro_tieback_list)
        locintro = locintro_tieback["location_intro"]
        tiebackstatement = locintro_tieback["tieback_question"]
        locname = random.choice(location_names)
        philsta = random.choice(philosophy_statements)
        fooddesc = random.choice(food_descriptions)
        mathp = random.choice(math_problems)
        used_locations.append(locname)
        
        sentence = f"{locintro} {locname}. {philsta}. {fooddesc}. {mathp}. {tiebackstatement}: {locname}"
        all_samples.append(sentence)

    # Number of generated samples
    num_samples = len(all_samples)

    # Compute sentence lengths (word-level, or use any tokenizer you prefer)
    sentence_lengths = [len(nltk.word_tokenize(s)) for s in all_samples]

    avg_length = statistics.mean(sentence_lengths)
    median_length = statistics.median(sentence_lengths)

    # Unique locations
    unique_locations = set(used_locations)
    num_unique_locations = len(unique_locations)

    # Top-5 most frequent locations
    loc_counter = Counter(used_locations)
    top5_locations = loc_counter.most_common(5)

    # Print or return a small table of stats
    print("="*50)
    print("Synthetic Co-reference Dataset Statistics")
    print("="*50)
    print(f"Number of Samples:                   {num_samples}")
    print(f"Average Tokens per Sample:           {avg_length:.2f}")
    print(f"Median Tokens per Sample:            {median_length}")
    print(f"Number of Unique Location Names:     {num_unique_locations}")
    print("Top-5 Most Frequent Locations:")
    for loc, freq in top5_locations:
        print(f"   {loc[:25]:<25} : {freq}")
    print("="*50)

if __name__ == "__main__":
    dataset_statistics(num_generations=10000, data_file="mrcr_custom.json")


# ==================================================
# Synthetic Co-reference Dataset Statistics
# ==================================================
# Number of Samples:                   10000
# Average Tokens per Sample:           250.95
# Median Tokens per Sample:            250.0
# Number of Unique Location Names:     96
# Top-5 Most Frequent Locations:
#    frostwhisper              : 326
#    eldershadow               : 232
#    glimmerforge              : 195
#    verdancleave              : 121
#    valerunth,                : 119
# ==================================================