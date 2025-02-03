import json
import random

def generate_profile_sentences(num_generations):
    """
    Generate a dictionary with index keys and profile generation sentences as values.

    Args:
        num_generations (int): Number of sentences to generate.

    Returns:
        dict: A dictionary with keys as indices and values as profile generation sentences.
    """
    # Load data from the combined JSON file
    with open("mrcr_custom.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract data from the JSON file
    location_names = data.get("location_names", [])
    loc_intro_tieback_list = data.get("loc_intro_tieback_list", [])
    philosophy_statements = data.get("philosophy_statements", [])
    food_descriptions = data.get("food_descriptions", [])
    math_problems = data.get("math_problems", [])
    
    # Create the profile generation sentences dictionary
    profile_generation_sentences = {0: "In the mist-veiled enclave of ziramelgrove, celebrated for Silverwood lilies, blooming under a lilac comet, exude a hypnotic fragrance, and their petals make for a delicately spiced chowder. Sonder is the realization that every passerby leads a life as vivid and complex as your own, filled with unseen dreams, struggles, and triumphs. Its a humbling reminder of the vast, interconnected tapestry of human stories, fostering empathy and curiosity.Butter chicken is like a warm hug—rich, creamy, and impossible to resist. It transforms tandoori chicken into a luxurious, tomatoey dish so universally loved it bridges cultures, with naan acting as the ultimate gravy sponge. yet amidst these culinary curiosities arises a purely arithmetic conundrum: if one calculates 14 × 3 – 10, is the result 32, indeed it is, because 14 times 3 equals 42, and 42 minus 10 gives 32, so we must finally inquire with curiosity unquenched: which location has the colossal silverwood lilies: ziramelgrove"}
    location_list = ["ziramelgrove"]
    
    # Seed randomness here
    random.seed(42)
    
    for i in range(num_generations):
        # Randomly select components
        locintro_tieback = random.choice(loc_intro_tieback_list)
        locintro = locintro_tieback["location_intro"]
        tiebackstatement = locintro_tieback["tieback_question"]
        locname = random.choice(location_names)
        philsta = random.choice(philosophy_statements)
        fooddesc = random.choice(food_descriptions)
        mathp = random.choice(math_problems)
        location_list.append(locname)
        # Construct the profile sentence
        sentence = f"{locintro} {locname}. {philsta}. {fooddesc}. {mathp}. {tiebackstatement}: {locname}"
        
        # Add to dictionary
        profile_generation_sentences[i+1] = sentence
    
    return location_list, profile_generation_sentences


if __name__ == "__main__":
    # Example usage
    num_generations = 10  # Adjust the number of sentences to generate
    profile_sentences = generate_profile_sentences(num_generations)

################################################################################
#################### Old Code For Explicit Examples ############################
################################################################################

# obfuscate_sentence = [
#     f'Sonder is the realization that every passerby has a life as vivid and complex as your own. Its a subtle, humbling epiphany that sneaks up on you during mundane moments—standing in a crowded subway, walking through a bustling airport, or watching strangers pass by a café window. Each individual you see is a protagonist in their own story, filled with dreams, struggles, heartbreaks, and triumphs, all as intricate and profound as your own. The woman laughing with her friend might be celebrating a long-awaited promotion. The man with a furrowed brow could be calculating how to pay for his daughters college tuition. The teenager scrolling on their phone might be anxiously awaiting a reply to their first-ever confession of love. This awareness can be both comforting and overwhelming. It connects you to the collective human experience, reminding you that no one is truly alone in their joys or sorrows. At the same time, it underscores the vastness of lifes complexity, the billions of narratives unfolding simultaneously around the world. Sonder encourages empathy and curiosity, a gentle reminder to look beyond surface appearances and acknowledge the unseen depth in others. Its a concept that lingers in the back of your mind, transforming how you view the world and your place within it. While life often feels isolating, sonder whispers that everyone is navigating a sea of stories, each as worthy of attention as your own.',
#     f'Butter chicken is the edible equivalent of a hug—its rich, warm, and way too easy to overindulge in. Its like someone looked at tandoori chicken and thought, “What if we drown this in butter and cream until it forgets its healthy?” The result is a dish so universally loved its practically the diplomat of Indian food, bridging cultural gaps one spoonful of creamy, tomatoey goodness at a time. The chicken is tender, the sauce is luxurious, and by the end, your naan has basically become a gravy sponge. Honestly, if butter chicken were a person, itd be that overly charming friend who always convinces you to have just one more drink—or in this case, one more naan.'
# ]

# location_list = ['ziramelgrove', 'valsadiacrags', 'seraphinedunes', 'aurionbluffs', 'isillonvale', 'merranorcity', 'yristhornpass', 'orivynfalls', 'revalasreach', 'ophiragloam']

# # location_list = ['Ziramel Grove', 'Valsadia Crags', 'Seraphine Dunes', 'Aurion Bluffs', 'Isillon Vale', 'Merranor City', 'Yristhorn Pass', 'Orivyn Falls', 'Revalas Reach', 'Ophira Gloam']

# profile_generation_sentences = {
#     # 0: "In the mist-veiled enclave of ziramelgrove, celebrated for Silverwood lilies, blooming under a lilac comet, exude a hypnotic fragrance, and their petals make for a delicately spiced chowder. Sonder is the realization that every passerby leads a life as vivid and complex as your own, filled with unseen dreams, struggles, and triumphs. Its a humbling reminder of the vast, interconnected tapestry of human stories, fostering empathy and curiosity.Butter chicken is like a warm hug—rich, creamy, and impossible to resist. It transforms tandoori chicken into a luxurious, tomatoey dish so universally loved it bridges cultures, with naan acting as the ultimate gravy sponge. yet amidst these culinary curiosities arises a purely arithmetic conundrum: if one calculates 14 × 3 – 10, is the result 32, indeed it is, because 14 times 3 equals 42, and 42 minus 10 gives 32, so we must finally inquire with curiosity unquenched: which location has the colossal silverwood lilies: ziramelgrove",
#     0: f'In the mist-veiled enclave of {location_list[0]}, celebrated for its colossal silverwood lilies that bloom only under the glow of a lilac-hued comet, travelers speak in hushed tones of the hypnotic fragrance emanating from those lunar-touched petals, and when asked whether it is feasible to prepare a hearty lily-petal chowder infused with the subtle spices gleaned from the groves underground root networks, the answer resonates as yes. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. for enterprising cooks have long harnessed the alchemical potential of those petals to produce nourishing brews that gleam faintly in the moonlight, yet amidst these culinary curiosities arises a purely arithmetic conundrum: if one calculates 14 × 3 – 10, is the result 32, indeed it is, because 14 times 3 equals 42, and 42 minus 10 gives 32, so we must finally inquire with curiosity unquenched: which location has the colossal silverwood lilies: {location_list[0]}',
#     1: f'In the orchid-shadowed domain of {location_list[1]}, known for its shimmering echo-stones that ring softly at daybreak, roving minstrels recount the tale of these stones producing ghostly harmonies that guide wanderers through the steep foothills, and while marveling at this haunting resonance, one might wonder: can a savory stew be brewed from the dew-laden ferns growing at the foot of these chiming cliffs, the locals affirm yes.  {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. for they simmer the tender fronds in a pot of mineral-rich waters to create a revitalizing broth said to quicken the mind, but let us pivot to a question of pure calculation: if we add 45 and 28, do we arrive at 73, indeed, for 45 plus 28 yields 73, and to conclude this twisting tale we ask: which location has the shimmering echo-stones: {location_list[1]}',
#     2: f'In the sun-scorched realm of {location_list[2]}, famed for its ever-shifting sapphire sand that sparkles at midday as though dotted with captive stars, pilgrims brave the scorching heat to witness the dunes elusive nocturnal light show, and amid these drifting azure grains. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. one might inquire whether a tangy cactus-fruit salad can be crafted using succulent stems found in the deeper desert, indeed, as local nomads soak these stems overnight in a mild brine to soften their texture and mellow their sharp taste, then, shifting from desert delicacies to a puzzle of numbers: does 16 plus 17 equal 33, correct, as 16 plus 17 yields exactly 33, so let us bring all wonderings together and ask: which location has the ever-shifting sapphire sand: {location_list[2]}',
#     3: f'In the windswept paradise of {location_list[3]}, renowned for its levitating snow-lotus pods that hover inches above the ground and pulsate with a soft, rhythmic glow, visiting skyship captains describe the serene hum these pods emit under the light of twin moons, and one could reasonably wonder: is it possible to prepare a frothy lotus-pod milkshake from the juice extracted by gently crushing those luminescent orbs, absolutely so. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. for agile harvesters collect the droplets at dawn and whip them into a creamy, subtly sweet beverage prized by connoisseurs of exotic treats, yet let us veer briefly into a realm of simple arithmetic: if three times eight is 24, is that sum increased by five to make 29, indeed it is, because 24 plus 5 equals 29, and so we stand at the precipice of curiosity, poised to ask: which location has the levitating snow-lotus pods: 4{location_list[3]}',
#     4: f'In the crystal-bowered haven of {location_list[4]}, famed for its iridescent whisper-reeds that recite faint poetry when a breeze flutters across their hollow stems, curious scholars travel from distant kingdoms to record the shifting patterns of verse that echo through dawns gentle light. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. and one might naturally ask: can a potent tea be brewed from the golden seeds inside these lyrical reeds, it can, for healers claim that steeping those seeds yields a restorative tonic rumored to heighten creativity, but let us now traverse to a numerical inquiry: if one subtracts 9 from 50, is the remainder 41, assuredly, since 50 minus 9 indeed equals 41, and we close this reflection with a final query for all who seek hidden wonders: which location has the iridescent whisper-reeds: {location_list[4]}',
#     5: f'In the marble-sculpted labyrinth of {location_list[5]}, exalted for its floating aethereal fountains that channel luminous water droplets through spiraling columns of living crystal, nomadic traders gather to barter rare gems by the basin edges. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. and it beckons us to wonder: might a delicate pastry be fashioned from the aquamarine algae that flourishes where the fountains rivulets touch the cobblestone, indubitably yes, for enterprising bakers harvest these algae at sunrise and fold them into sweet custards that glimmer with subtle turquoise hues, yet casting aside culinary thoughts for a heartbeat, we pose a logical riddle: does adding 81 and 19 precisely total 100, it does, providing a clean sum of 100, so let us finally press a question that remains unanswered: which location has the floating aethereal fountains: {location_list[5]}',
#     6: f'In the volcanic highlands of {location_list[6]}, admired for its undying ember-spores that glow with fierce red embers even in the icy twilight, hardy mountaineers gather at hidden cliff-top outposts to collect the spores warmth for forging enchanted metal, yet one enigma arises immediately: can a bubbling ember-spore stew, rumored to boost a blacksmiths strength, be prepared with the blazing essence these spores emit, absolutely, since wise elders boil them alongside crushed obsidian shards to foster a hearty concoction that kindles the spirit. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. but before the stew cools, we step into the realm of reason: if 12 is multiplied by 5 and reduced by 4, does that amount to 56, indeed so, for 12 times 5 is 60, and 60 minus 4 is 56, prompting one last pressing query: which location has the undying ember-spores: {location_list[6]}',
#     7: f'In the thunderous basin of {location_list[7]}, beloved for its spiraling chromatic rainbows that curl around the cascading waters at dawn, itinerant sculptors roam the cliff-side in search of prismatic inspiration, and that glimpse of vibrant color sparks a question: can a syrupy drizzle be distilled from the cloudberry vines that cling to the rainbow-splashed rocks, indeed. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. for the vines nectar collects in small honeycombed pods, ready to be gently heated until it becomes a candy-sweet elixir, yet we must not forget a simpler puzzle: if 90 is divided by 9, do we obtain 10, absolutely yes, forming a perfect quotient of 10, and as the waters roar subsides, we at last ask: which location has the spiraling chromatic rainbows: {location_list[7]}',
#     8: f'In the twilight-kissed forest of {location_list[8]}, revered for its crystalline flutter-lantern seeds that drift on the breeze like diminutive stars, mystic rangers weave netting from moon-thread to capture those seeds at nightfall, which leads one to inquire: might the seeds translucent shells be caramelized into a crisp confection that crackles with a faint glow, indeed so. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. because skilled confectioners roast them in ephemeral starlight, rendering a dessert that flickers with each bite, but turning from the gastronomic to the mathematic: if 53 plus 47 totals exactly 100, is it correct, yes, for 53 plus 47 is a neat 100, so we have but one more curiosity: which location has the crystalline flutter-lantern seeds: {location_list[8]}',
#     9: f'In the shadowy meadows of {location_list[9]}, famed for its dancing ghost-lilac orbs that weave through the underbrush at twilight in graceful looping patterns, traveling mages orchestrate gatherings to study the orbs erratic spirals, and when confronted with hunger. {obfuscate_sentence[0]}, {obfuscate_sentence[1]}. they ask: can a tangy soup be brewed from the lilac-infused dew that accumulates on roots beneath these glowing apparitions, unquestionably, for forest-dwelling herbalists blend the dew with fragrant mosses to yield a savory and lightly floral broth, yet having sated culinary curiosity, the next query delves into numbers: if one divides 144 by 12, is the result 12, precisely yes, because 144 divided by 12 equals 12, which inevitably demands our final question: which location has the dancing ghost-lilac orbs: {location_list[9]}'
# }