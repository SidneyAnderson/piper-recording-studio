"""Sentence templates for generating phonetically diverse TTS training prompts.

Each category contains templates with slots that get filled with varied vocabulary.
Templates are designed to:
- Cover all English phonemes through natural word choices
- Produce varied prosodic patterns (statements, questions, commands, etc.)
- Mix British and General English phrasing
- Vary sentence length from short (4 words) to long (25+ words)
"""

import random

# ============================================================================
# Vocabulary pools — drawn from to fill template slots
# ============================================================================

NAMES_BRITISH = [
    "Oliver", "Amelia", "George", "Charlotte", "Harry", "Isla", "Jack",
    "Emily", "Thomas", "Sophie", "William", "Poppy", "James", "Mia",
    "Edward", "Freya", "Arthur", "Florence", "Henry", "Grace", "Eleanor",
    "Alfie", "Rosie", "Archie", "Evie", "Oscar", "Lily", "Leo", "Ivy",
    "Freddie", "Ella", "Charlie", "Scarlett", "Theodore", "Phoebe",
]

CITIES_BRITISH = [
    "London", "Edinburgh", "Manchester", "Bristol", "Oxford", "Cambridge",
    "Birmingham", "Liverpool", "Glasgow", "Cardiff", "Belfast", "Bath",
    "Brighton", "York", "Canterbury", "Winchester", "Stratford", "Dover",
    "Aberdeen", "Inverness", "Swansea", "Norwich", "Exeter", "Durham",
]

CITIES_GENERAL = [
    "New York", "Paris", "Tokyo", "Sydney", "Toronto", "Berlin",
    "Barcelona", "Amsterdam", "Vienna", "Prague", "Dublin", "Rome",
    "Stockholm", "Helsinki", "Lisbon", "Athens", "Seoul", "Singapore",
]

FOODS = [
    "pasta", "salmon", "chicken", "rice", "bread", "cheese", "soup",
    "salad", "potatoes", "mushrooms", "tomatoes", "chocolate", "biscuits",
    "scones", "porridge", "crumpets", "marmite", "shepherd's pie",
    "fish and chips", "bangers and mash", "Yorkshire pudding", "treacle tart",
    "strawberries", "roast beef", "lamb chops", "curry", "sushi", "tacos",
]

COLOURS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "white",
    "black", "grey", "brown", "turquoise", "crimson", "ivory", "navy",
    "silver", "golden", "scarlet", "emerald", "sapphire", "amber", "coral",
]

WEATHER = [
    "sunny", "cloudy", "rainy", "foggy", "windy", "stormy", "overcast",
    "drizzly", "misty", "clear", "humid", "chilly", "freezing", "mild",
    "warm", "breezy", "thundery", "frosty", "icy", "balmy", "muggy",
]

JOBS = [
    "teacher", "doctor", "engineer", "chef", "architect", "solicitor",
    "journalist", "pharmacist", "librarian", "plumber", "electrician",
    "accountant", "designer", "musician", "photographer", "barrister",
    "surgeon", "veterinarian", "consultant", "researcher", "pilot",
]

TRANSPORT = [
    "bus", "train", "underground", "taxi", "bicycle", "car", "ferry",
    "aeroplane", "tram", "coach", "lorry", "motorbike", "helicopter",
]

ROOMS = [
    "kitchen", "bedroom", "living room", "bathroom", "garden", "study",
    "dining room", "conservatory", "garage", "attic", "cellar", "hallway",
]

ADJECTIVES = [
    "brilliant", "lovely", "gorgeous", "dreadful", "splendid", "awful",
    "magnificent", "terrible", "wonderful", "ghastly", "fabulous",
    "frightful", "charming", "tedious", "exceptional", "appalling",
    "delightful", "hideous", "remarkable", "peculiar", "superb",
    "outstanding", "moderate", "substantial", "considerable", "significant",
    "extraordinary", "traditional", "contemporary", "sophisticated",
]

MATERIALS = [
    "wooden", "leather", "cotton", "silk", "steel", "glass", "ceramic",
    "marble", "bronze", "copper", "velvet", "linen", "stone", "concrete",
]

TIMES = [
    "half past three", "quarter to seven", "twenty past nine",
    "ten to twelve", "five past two", "quarter past eight",
    "half past six", "twenty to four", "noon", "midnight",
    "early morning", "late afternoon", "early evening", "just before dawn",
]

NUMBERS_SPOKEN = [
    "forty-seven", "three hundred and twelve", "one thousand two hundred",
    "sixty-five point three", "nineteen eighty-four", "two thousand and twenty-six",
    "fourteen million", "nought point seven five", "seven eighths",
    "three quarters", "two thirds", "one hundred and fifty-six",
    "eight thousand nine hundred", "twelve point five per cent",
]

BRITISH_EXPRESSIONS = [
    "straightaway", "fortnight", "brilliant", "cheers", "queueing",
    "rubbish bin", "car park", "cinema", "motorway", "petrol station",
    "postbox", "chemist", "nappy", "crisps", "lift", "flat",
    "boot of the car", "bonnet", "pavement", "zebra crossing",
    "roundabout", "biscuit tin", "washing up", "having a lie in",
]

EMOTIONS = [
    "excited", "worried", "delighted", "frustrated", "grateful",
    "anxious", "relieved", "disappointed", "thrilled", "concerned",
    "overwhelmed", "confident", "nervous", "content", "furious",
    "astonished", "bewildered", "enthusiastic", "melancholic", "optimistic",
]

ANIMALS = [
    "hedgehog", "robin", "badger", "squirrel", "fox", "pheasant",
    "swan", "otter", "deer", "hare", "owl", "raven", "dolphin",
    "elephant", "penguin", "giraffe", "cheetah", "gorilla", "octopus",
]

HOBBIES = [
    "gardening", "painting", "knitting", "baking", "reading", "hiking",
    "photography", "fishing", "woodworking", "pottery", "cycling",
    "swimming", "yoga", "chess", "birdwatching", "calligraphy",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "twenty-first", "thirtieth", "thirty-first",
]


def _pick(pool, n=1):
    """Pick n unique random items from a pool."""
    return random.sample(pool, min(n, len(pool)))


def _p(pool):
    """Pick one random item from a pool."""
    return random.choice(pool)


# ============================================================================
# Template functions — each returns a single generated sentence
# ============================================================================

def daily_life():
    """Generate a sentence about everyday routines, errands, and household life."""
    templates = [
        f"{_p(NAMES_BRITISH)} usually wakes up at {_p(TIMES)} and starts the day with a cup of tea.",
        f"Could you please pass me the {_p(COLOURS)} {_p(MATERIALS)} box from the {_p(ROOMS)}?",
        f"The {_p(ADJECTIVES)} weather today means we should probably stay indoors.",
        f"I need to pop to the shops {_p(BRITISH_EXPRESSIONS)} to pick up some {_p(FOODS)}.",
        f"Have you seen my {_p(COLOURS)} jacket? I left it in the {_p(ROOMS)} yesterday.",
        f"We ought to leave by {_p(TIMES)} if we want to avoid the traffic.",
        f"The {_p(JOBS)} said the appointment would take about {_p(NUMBERS_SPOKEN)} minutes.",
        f"{_p(NAMES_BRITISH)} felt {_p(EMOTIONS)} when the parcel finally arrived.",
        f"I'm not sure whether to take the {_p(TRANSPORT)} or walk to the {_p(BRITISH_EXPRESSIONS)}.",
        f"It's been a {_p(ADJECTIVES)} week, hasn't it?",
        f"There's a {_p(ADJECTIVES)} little cafe just round the corner from the {_p(BRITISH_EXPRESSIONS)}.",
        f"Would you mind closing the window in the {_p(ROOMS)}? It's getting rather {_p(WEATHER)}.",
        f"She always keeps her {_p(MATERIALS)} bag in the {_p(ROOMS)} next to the door.",
    ]
    return random.choice(templates)


def weather_nature():
    """Generate a sentence about weather conditions, seasons, or wildlife."""
    templates = [
        f"The forecast says it'll be {_p(WEATHER)} tomorrow with temperatures around {_p(NUMBERS_SPOKEN)} degrees.",
        f"What a {_p(ADJECTIVES)} {_p(WEATHER)} morning! Perfect for a walk in the park.",
        f"The {_p(ANIMALS)} appeared in the garden just as the rain stopped.",
        f"I don't think we've had such {_p(WEATHER)} weather since last {_p(MONTHS)}.",
        f"Look at those {_p(ADJECTIVES)} {_p(COLOURS)} clouds gathering over the hills.",
        f"The {_p(WEATHER)} conditions made driving on the {_p(BRITISH_EXPRESSIONS)} quite difficult.",
        f"Spring brings the most {_p(ADJECTIVES)} wildflowers to the countryside.",
        f"Can you hear the {_p(ANIMALS)} singing? It must be nearly {_p(TIMES)}.",
        f"The river was {_p(ADJECTIVES)} after three days of {_p(WEATHER)} weather.",
        f"In {_p(MONTHS)}, the {_p(ANIMALS)} migrate south for the winter.",
        f"The garden looks {_p(ADJECTIVES)} this time of year, especially the {_p(COLOURS)} roses.",
    ]
    return random.choice(templates)


def food_cooking():
    """Generate a sentence about cooking, recipes, or dining."""
    templates = [
        f"Could you stir the {_p(FOODS)} while I prepare the {_p(FOODS)}?",
        f"This recipe calls for {_p(NUMBERS_SPOKEN)} grams of {_p(FOODS)} and a pinch of salt.",
        f"The {_p(ADJECTIVES)} {_p(FOODS)} at that restaurant in {_p(CITIES_BRITISH)} was unforgettable.",
        f"I prefer my {_p(FOODS)} with a bit of {_p(FOODS)} on the side.",
        f"Have you ever tried making {_p(FOODS)} from scratch? It's {_p(ADJECTIVES)}!",
        f"We need to pick up some fresh {_p(FOODS)} from the market before {_p(TIMES)}.",
        f"The {_p(JOBS)} recommended pairing the {_p(FOODS)} with a glass of red wine.",
        f"{_p(NAMES_BRITISH)} baked {_p(ADJECTIVES)} {_p(FOODS)} for the village fete.",
        f"Is the {_p(FOODS)} ready yet? I'm absolutely starving.",
        f"My grandmother's {_p(FOODS)} recipe has been in the family for generations.",
        f"You'll want to let the {_p(FOODS)} cool for about {_p(NUMBERS_SPOKEN)} minutes before serving.",
    ]
    return random.choice(templates)


def travel_directions():
    """Generate a sentence about journeys, navigation, or public transport."""
    templates = [
        f"The fastest way to {_p(CITIES_BRITISH)} is by {_p(TRANSPORT)} from the central station.",
        f"Turn left at the {_p(BRITISH_EXPRESSIONS)} and continue for about {_p(NUMBERS_SPOKEN)} metres.",
        f"How long does it take to get from {_p(CITIES_BRITISH)} to {_p(CITIES_BRITISH)} by {_p(TRANSPORT)}?",
        f"The {_p(TRANSPORT)} to {_p(CITIES_GENERAL)} departs at {_p(TIMES)} from platform six.",
        f"We visited {_p(CITIES_GENERAL)} last {_p(MONTHS)} and the weather was {_p(ADJECTIVES)}.",
        f"Could you tell me where the nearest {_p(BRITISH_EXPRESSIONS)} is, please?",
        f"Take the {_p(ORDINALS)} exit at the {_p(BRITISH_EXPRESSIONS)} and follow signs to {_p(CITIES_BRITISH)}.",
        f"The journey from {_p(CITIES_BRITISH)} takes approximately {_p(NUMBERS_SPOKEN)} hours.",
        f"I'd recommend booking the {_p(TRANSPORT)} tickets in advance, especially in {_p(MONTHS)}.",
        f"Is this the right platform for the {_p(TRANSPORT)} to {_p(CITIES_BRITISH)}?",
        f"We got completely lost trying to find the {_p(BRITISH_EXPRESSIONS)} near {_p(CITIES_BRITISH)}.",
    ]
    return random.choice(templates)


def technology():
    """Generate a sentence about software, devices, or digital systems."""
    templates = [
        f"The new software update should fix the {_p(ADJECTIVES)} performance issues we've been having.",
        f"Have you tried restarting the device? That often resolves the problem {_p(BRITISH_EXPRESSIONS)}.",
        f"The algorithm processes approximately {_p(NUMBERS_SPOKEN)} requests per second.",
        f"I think the {_p(COLOURS)} screen issue might be related to the display settings.",
        f"The {_p(JOBS)} explained that the system would be offline for about {_p(NUMBERS_SPOKEN)} hours.",
        f"Could you send me the document by email before {_p(TIMES)}?",
        f"The battery lasts roughly {_p(NUMBERS_SPOKEN)} hours under normal usage conditions.",
        f"This {_p(ADJECTIVES)} application was developed by a team in {_p(CITIES_GENERAL)}.",
        f"You'll need to enter your password and then click the {_p(COLOURS)} button.",
        f"The database contains over {_p(NUMBERS_SPOKEN)} records from the past decade.",
    ]
    return random.choice(templates)


def health_wellness():
    """Generate a sentence about exercise, medical appointments, or wellbeing."""
    templates = [
        f"The {_p(JOBS)} advised getting at least {_p(NUMBERS_SPOKEN)} minutes of exercise daily.",
        f"I've been feeling rather {_p(EMOTIONS)} since I started {_p(HOBBIES)} every morning.",
        f"Please remember to take your medication at {_p(TIMES)} with a glass of water.",
        f"The appointment is scheduled for the {_p(ORDINALS)} of {_p(MONTHS)} at {_p(TIMES)}.",
        f"Walking is one of the most {_p(ADJECTIVES)} forms of exercise for overall health.",
        f"She felt {_p(EMOTIONS)} after finishing her first marathon in {_p(CITIES_BRITISH)}.",
        f"The {_p(JOBS)} recommended drinking {_p(NUMBERS_SPOKEN)} litres of water each day.",
        f"Getting enough sleep is {_p(ADJECTIVES)} important for your mental wellbeing.",
        f"Have you noticed any improvement since you started the new treatment?",
        f"The waiting room was {_p(ADJECTIVES)} quiet for a Monday morning.",
    ]
    return random.choice(templates)


def sports_hobbies():
    """Generate a sentence about sports matches, leisure activities, or hobbies."""
    templates = [
        f"{_p(NAMES_BRITISH)} has been practising {_p(HOBBIES)} for about {_p(NUMBERS_SPOKEN)} years.",
        f"The match between {_p(CITIES_BRITISH)} and {_p(CITIES_BRITISH)} was absolutely {_p(ADJECTIVES)}!",
        f"I took up {_p(HOBBIES)} during the lockdown and haven't stopped since.",
        f"The {_p(ADJECTIVES)} thing about {_p(HOBBIES)} is that anyone can learn it.",
        f"We scored in the {_p(ORDINALS)} minute, which was {_p(ADJECTIVES)} timing.",
        f"Have you ever tried {_p(HOBBIES)}? It's more {_p(ADJECTIVES)} than you might think.",
        f"The {_p(HOBBIES)} class starts at {_p(TIMES)} every Saturday at the community centre.",
        f"She won {_p(ORDINALS)} place in the national {_p(HOBBIES)} competition in {_p(CITIES_BRITISH)}.",
        f"I find {_p(HOBBIES)} {_p(ADJECTIVES)} relaxing after a long day at work.",
        f"The {_p(COLOURS)} team defeated the {_p(COLOURS)} team by {_p(NUMBERS_SPOKEN)} points.",
    ]
    return random.choice(templates)


def education_learning():
    """Generate a sentence about school, university, or studying."""
    templates = [
        f"The {_p(JOBS)} explained the concept in a {_p(ADJECTIVES)} clear and engaging way.",
        f"{_p(NAMES_BRITISH)} is studying at the University of {_p(CITIES_BRITISH)} this year.",
        f"The exam results will be published on the {_p(ORDINALS)} of {_p(MONTHS)}.",
        f"Could you recommend a {_p(ADJECTIVES)} book on the subject?",
        f"The lecture covered approximately {_p(NUMBERS_SPOKEN)} different topics in two hours.",
        f"I've always found {_p(HOBBIES)} to be the most {_p(ADJECTIVES)} subject at school.",
        f"The library closes at {_p(TIMES)}, so we need to return the books before then.",
        f"Children learn best when they feel {_p(EMOTIONS)} and supported in the classroom.",
        f"The course runs for {_p(NUMBERS_SPOKEN)} weeks starting in {_p(MONTHS)}.",
        f"The {_p(ADJECTIVES)} presentation received a standing ovation from the audience.",
    ]
    return random.choice(templates)


def business_finance():
    """Generate a sentence about meetings, reports, or office work."""
    templates = [
        f"The quarterly report shows revenue increased by {_p(NUMBERS_SPOKEN)} per cent.",
        f"We need to submit the proposal to the {_p(JOBS)} before {_p(TIMES)} on Friday.",
        f"The meeting has been rescheduled to the {_p(ORDINALS)} of {_p(MONTHS)}.",
        f"Our {_p(CITIES_BRITISH)} office will handle the {_p(ADJECTIVES)} project going forward.",
        f"The total cost came to approximately {_p(NUMBERS_SPOKEN)} pounds including tax.",
        f"Could you prepare a {_p(ADJECTIVES)} summary of the findings by tomorrow?",
        f"The company employs over {_p(NUMBERS_SPOKEN)} people across the United Kingdom.",
        f"I'll send the invoice {_p(BRITISH_EXPRESSIONS)} once the work is completed.",
        f"{_p(NAMES_BRITISH)} was appointed as the new managing director last {_p(MONTHS)}.",
        f"The contract stipulates a {_p(NUMBERS_SPOKEN)} day notice period for termination.",
    ]
    return random.choice(templates)


def arts_entertainment():
    """Generate a sentence about galleries, films, music, or performances."""
    templates = [
        f"The {_p(ADJECTIVES)} exhibition at the gallery in {_p(CITIES_BRITISH)} runs until {_p(MONTHS)}.",
        f"Have you seen the new film? The reviews say it's absolutely {_p(ADJECTIVES)}.",
        f"{_p(NAMES_BRITISH)} performed a {_p(ADJECTIVES)} rendition of the {_p(ORDINALS)} symphony.",
        f"The theatre production received {_p(ADJECTIVES)} reviews from critics and audiences alike.",
        f"Tickets for the concert in {_p(CITIES_BRITISH)} go on sale at {_p(TIMES)} tomorrow.",
        f"The {_p(COLOURS)} painting in the corner is by a local artist from {_p(CITIES_BRITISH)}.",
        f"I'd give the performance {_p(NUMBERS_SPOKEN)} out of ten, personally.",
        f"The museum opens at {_p(TIMES)} and admission costs {_p(NUMBERS_SPOKEN)} pounds.",
        f"She's been playing the piano since she was about {_p(NUMBERS_SPOKEN)} years old.",
        f"The {_p(ADJECTIVES)} documentary explores life in {_p(CITIES_GENERAL)} during the last century.",
    ]
    return random.choice(templates)


def science():
    """Generate a sentence about experiments, research, or natural phenomena."""
    templates = [
        f"The experiment yielded {_p(ADJECTIVES)} results that exceeded our initial expectations.",
        f"Researchers at the University of {_p(CITIES_BRITISH)} published their findings last {_p(MONTHS)}.",
        f"The temperature must remain below {_p(NUMBERS_SPOKEN)} degrees for the reaction to work.",
        f"This {_p(ADJECTIVES)} discovery could change the way we understand the natural world.",
        f"The {_p(ANIMALS)} population has declined by approximately {_p(NUMBERS_SPOKEN)} per cent.",
        f"The chemical formula requires precisely {_p(NUMBERS_SPOKEN)} milligrams of the compound.",
        f"According to the study, {_p(NUMBERS_SPOKEN)} per cent of participants showed improvement.",
        f"The {_p(ADJECTIVES)} telescope can observe objects {_p(NUMBERS_SPOKEN)} light years away.",
        f"Climate data from the past {_p(NUMBERS_SPOKEN)} years reveals a clear upward trend.",
        f"The {_p(JOBS)} explained the hypothesis in terms anyone could understand.",
    ]
    return random.choice(templates)


def history_culture():
    """Generate a sentence about historical events, traditions, or heritage sites."""
    templates = [
        f"The castle was built in the {_p(ORDINALS)} century and has been restored beautifully.",
        f"{_p(CITIES_BRITISH)} has a rich history dating back over {_p(NUMBERS_SPOKEN)} years.",
        f"The {_p(ADJECTIVES)} tradition of afternoon tea originated in the eighteen forties.",
        f"This {_p(MATERIALS)} artefact was discovered near {_p(CITIES_BRITISH)} in {_p(MONTHS)} last year.",
        f"The {_p(ORDINALS)} king of that name ruled for approximately {_p(NUMBERS_SPOKEN)} years.",
        f"Victorian architecture is particularly {_p(ADJECTIVES)} in the streets of {_p(CITIES_BRITISH)}.",
        f"The museum's {_p(ADJECTIVES)} collection includes over {_p(NUMBERS_SPOKEN)} historical items.",
        f"Many {_p(ADJECTIVES)} customs and traditions are still observed in the countryside.",
        f"The {_p(ADJECTIVES)} cathedral was consecrated on the {_p(ORDINALS)} of {_p(MONTHS)}.",
        f"The local dialect in {_p(CITIES_BRITISH)} has some {_p(ADJECTIVES)} unique expressions.",
    ]
    return random.choice(templates)


def emotions_social():
    """Generate a sentence expressing feelings or social interactions."""
    templates = [
        f"I'm absolutely {_p(EMOTIONS)} about the news! When did you find out?",
        f"{_p(NAMES_BRITISH)} seemed {_p(EMOTIONS)} when we told them about the surprise.",
        f"It's perfectly natural to feel {_p(EMOTIONS)} when facing such a big change.",
        f"The {_p(ADJECTIVES)} gesture made everyone in the room feel {_p(EMOTIONS)}.",
        f"How are you feeling today? You look a bit {_p(EMOTIONS)}.",
        f"I was so {_p(EMOTIONS)} that I couldn't find the words to express my gratitude.",
        f"The children were {_p(EMOTIONS)} to see the {_p(ANIMALS)} at the wildlife centre.",
        f"Don't you think it's {_p(ADJECTIVES)} how quickly things can change?",
        f"She tried to remain calm but was clearly feeling {_p(EMOTIONS)} inside.",
        f"Being {_p(EMOTIONS)} is nothing to be ashamed of, you know.",
    ]
    return random.choice(templates)


def home_garden():
    """Generate a sentence about interior decor, repairs, or gardening."""
    templates = [
        f"The {_p(COLOURS)} curtains in the {_p(ROOMS)} really brighten up the whole space.",
        f"We need to fix the leak in the {_p(ROOMS)} before it causes more damage.",
        f"The {_p(MATERIALS)} floor in the {_p(ROOMS)} needs to be polished this weekend.",
        f"I planted some {_p(COLOURS)} flowers in the garden last {_p(MONTHS)}.",
        f"The {_p(JOBS)} quoted {_p(NUMBERS_SPOKEN)} pounds for the repair work.",
        f"Could you help me move the {_p(MATERIALS)} table from the {_p(ROOMS)} to the garden?",
        f"We've been meaning to redecorate the {_p(ROOMS)} for about {_p(NUMBERS_SPOKEN)} months now.",
        f"The {_p(ANIMALS)} has been digging up the flower beds again.",
        f"I think the {_p(COLOURS)} paint would look {_p(ADJECTIVES)} on the {_p(ROOMS)} walls.",
        f"The house was originally built in the {_p(ORDINALS)} century and has {_p(MATERIALS)} beams.",
    ]
    return random.choice(templates)


def news_current_affairs():
    """Generate a sentence about government, policy, or public events."""
    templates = [
        f"The Prime Minister announced the new policy at {_p(TIMES)} this morning.",
        f"Residents of {_p(CITIES_BRITISH)} have expressed {_p(EMOTIONS)} views about the proposal.",
        f"The report estimates that approximately {_p(NUMBERS_SPOKEN)} people will be affected.",
        f"Opposition leaders described the decision as {_p(ADJECTIVES)} and short-sighted.",
        f"Emergency services responded to the incident within {_p(NUMBERS_SPOKEN)} minutes.",
        f"The {_p(ADJECTIVES)} debate in Parliament lasted for over {_p(NUMBERS_SPOKEN)} hours.",
        f"Transport disruptions are expected to continue until the {_p(ORDINALS)} of {_p(MONTHS)}.",
        f"The {_p(JOBS)} told reporters that further details would follow {_p(BRITISH_EXPRESSIONS)}.",
        f"Public opinion polls suggest that {_p(NUMBERS_SPOKEN)} per cent support the measure.",
        f"The council meeting in {_p(CITIES_BRITISH)} will address the issue next {_p(MONTHS)}.",
    ]
    return random.choice(templates)


def british_culture():
    """Generate a sentence using British idioms, customs, or cultural references."""
    templates = [
        f"Shall we pop into the {_p(BRITISH_EXPRESSIONS)} on the way home for some {_p(FOODS)}?",
        f"The queue at the {_p(BRITISH_EXPRESSIONS)} stretched right round the corner.",
        f"I reckon we'll need a {_p(COLOURS)} brolly today, judging by those clouds.",
        f"It's not very British to make a fuss, but this is {_p(ADJECTIVES)} unacceptable.",
        f"We spent a {_p(ADJECTIVES)} afternoon having a cream tea in {_p(CITIES_BRITISH)}.",
        f"The {_p(BRITISH_EXPRESSIONS)} was closed for refurbishment until {_p(MONTHS)}.",
        f"Right, I'll put the kettle on while you sort out the {_p(ROOMS)}.",
        f"The village fete raised over {_p(NUMBERS_SPOKEN)} pounds for the local charity.",
        f"Apparently the {_p(BRITISH_EXPRESSIONS)} near the high street is shutting down.",
        f"Cheers for helping out yesterday, {_p(NAMES_BRITISH)}. Much appreciated.",
    ]
    return random.choice(templates)


def numbers_dates_times():
    """Generate a sentence rich in spoken numbers, dates, and times."""
    templates = [
        f"The train departs at {_p(TIMES)} from platform {_p(ORDINALS).split('-')[0]}.",
        f"Her birthday is on the {_p(ORDINALS)} of {_p(MONTHS)}, nineteen ninety-seven.",
        f"The total comes to {_p(NUMBERS_SPOKEN)} pounds and {_p(NUMBERS_SPOKEN)} pence.",
        f"We've been living here since the {_p(ORDINALS)} of {_p(MONTHS)}, two thousand and twelve.",
        f"The building is approximately {_p(NUMBERS_SPOKEN)} metres tall and {_p(NUMBERS_SPOKEN)} metres wide.",
        f"There were about {_p(NUMBERS_SPOKEN)} people at the event last Saturday.",
        f"The temperature dropped to minus {_p(NUMBERS_SPOKEN)} degrees overnight.",
        f"Flight number {_p(NUMBERS_SPOKEN)} to {_p(CITIES_GENERAL)} boards at {_p(TIMES)}.",
        f"It weighs roughly {_p(NUMBERS_SPOKEN)} kilograms and measures {_p(NUMBERS_SPOKEN)} centimetres.",
        f"The recipe serves {_p(NUMBERS_SPOKEN)} people and takes {_p(NUMBERS_SPOKEN)} minutes to prepare.",
    ]
    return random.choice(templates)


def phone_addresses():
    """Generate a sentence containing phone numbers, postcodes, or street addresses."""
    n = lambda: str(random.randint(0, 9))
    templates = [
        f"You can reach me on oh seven {n()}{n()}{n()}, {n()}{n()}{n()}, {n()}{n()}{n()}.",
        f"The postcode is {_p(CITIES_BRITISH)[:2].upper()}{n()}{n()} {n()}{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}.",
        f"Our address is {_p(NUMBERS_SPOKEN)} {_p(COLOURS).title()} Street, {_p(CITIES_BRITISH)}.",
        f"Please send correspondence to flat {n()}{n()}, {_p(ADJECTIVES).title()} House, {_p(CITIES_BRITISH)}.",
        f"The dialling code for {_p(CITIES_BRITISH)} is oh one {n()}{n()}{n()}.",
        f"My reference number is {random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{n()}{n()}{n()}{n()}{n()}{n()}.",
        f"You can find us at number {_p(NUMBERS_SPOKEN)}, {_p(CITIES_BRITISH)} Road.",
        f"The emergency number is nine nine nine, or you can call one one two.",
    ]
    return random.choice(templates)


# Map category names to generator functions
GENERATORS = {
    "daily_life": daily_life,
    "weather_nature": weather_nature,
    "food_cooking": food_cooking,
    "travel_directions": travel_directions,
    "technology": technology,
    "health_wellness": health_wellness,
    "sports_hobbies": sports_hobbies,
    "education_learning": education_learning,
    "business_finance": business_finance,
    "arts_entertainment": arts_entertainment,
    "science": science,
    "history_culture": history_culture,
    "emotions_social": emotions_social,
    "home_garden": home_garden,
    "news_current_affairs": news_current_affairs,
    "british_culture": british_culture,
    "numbers_dates_times": numbers_dates_times,
    "phone_addresses": phone_addresses,
}
