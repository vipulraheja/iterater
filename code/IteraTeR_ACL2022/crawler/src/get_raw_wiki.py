import os
import json
import requests
import argparse
import mwparserfromhell
from utils import get_time_range


def get_data(start, end, gcmcontinue, gcmtitle):
    if len(gcmcontinue)>0:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": gcmtitle,
            "gcmsort": "timestamp",
            "gcmlimit": "100",
            "gcmdir": "desc",
            "gcmcontinue": gcmcontinue,
            "formatversion": "2",
            "format": "json",
        }
    else:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": gcmtitle,
            "gcmsort": "timestamp",
            "gcmlimit": "100",
            "gcmdir": "desc",
            "gcmstart": end,
            "formatversion": "2",
            "format": "json",
        }

    data = S.get(url=URL, params=params).json()
    try:
        pages = data["query"]["pages"]
    except:
        pages = []
        print(f'Cannot get {gcmtitle}!!!!!!!!')
    try:
        gcmcontinue = data["continue"]["gcmcontinue"]
    except:
        gcmcontinue = ''
    return pages, gcmcontinue

def get_subcategories(start, end, gcmcontinue, gcmtitle):
    if len(gcmcontinue)>0:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": gcmtitle,
            "cmlimit": "100",
            "cmtype": "subcat",
            "cmcontinue": gcmcontinue,
            "formatversion": "2",
            "format": "json",
        }
    else:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": gcmtitle,
            "cmlimit": "100",
            "cmtype": "subcat",
            "cmcontinue": gcmcontinue,
            "cmstart": end,
            "formatversion": "2",
            "format": "json",
        }

    data = S.get(url=URL, params=params).json()
    try:
        pages = data["query"]["categorymembers"]
    except:
        pages = []
        print(f'Cannot get {gcmtitle}!!!!!!!!')
    try:
        gcmcontinue = data["continue"]["cmcontinue"]
    except:
        gcmcontinue = ''
    return pages, gcmcontinue

def parse_subcate_list(start, end, gcmtitle):
    outs, cates = [], []
    gcmcontinue = ""
    pages, gcmcontinue = get_subcategories(start, end, gcmcontinue, gcmtitle)
    outs += pages
    while len(gcmcontinue)>0:
        pages, gcmcontinue = get_subcategories(start, end, gcmcontinue, gcmtitle)
        outs += pages
    for out in outs:
        cate = out['title'].replace(' ', "_")
        cate = cate.replace("Category:", "")
        cates.append(cate)
    return cates

def get_last_n_revisions(pageid, n):
    responses = S.get(url=URL, params={"action":"query", 
                                    "prop": "revisions",
                                    "pageids":f"{pageid}",
                                    "rvlimit": f"{n}",
                                    "formatversion": "2",
                                    "format": "json",}).json()
    revisions = responses["query"]["pages"][0]["revisions"]
    return revisions

def parse_revision(pages, gcmcontinue, json_file):
    out = []
    for page in pages:
        pageid = page["pageid"]
        title = page['title']
        revisions = get_last_n_revisions(pageid, 20)
        for rev in revisions:
            if rev['minor']: continue
            rev["pageid"] = pageid
            rev["title"] = title
            revid = rev["revid"]
            parentid = rev["parentid"]
            responses = S.get(url=URL, params={"action":"query", 
                                                  "prop": "revisions",
                                                  "revids":f"{revid}",
                                                  "rvslots": "main",
                                                  "rvprop": "content",
                                                  "formatversion": "2",
                                                  "format": "json",}).json()
            try:
                cur_content = responses["query"]["pages"][0]["revisions"][0]['slots']['main']['content']
                rev["cur_content"] = mwparserfromhell.parse(cur_content).strip_code()
            except:
                print(f'Fail to parse {revid}!!!!')
                continue
            responses = S.get(url=URL, params={"action":"query", 
                                                  "prop": "revisions",
                                                  "revids":f"{parentid}",
                                                  "rvslots": "main",
                                                  "rvprop": "content",
                                                  "formatversion": "2",
                                                  "format": "json",}).json()
            try:
                parent_content = responses["query"]["pages"][0]["revisions"][0]['slots']['main']['content']
                rev["parent_content"] = mwparserfromhell.parse(parent_content).strip_code()
            except:
                print(f'Fail to parse {parentid}!!!!')
                continue
            out.append(rev)
            json_file.write(json.dumps(rev)+'\n')
    return out

def get_wiki_cates_list(main_cate):
    if main_cate == 'philosophy':
        cates = ["Philosophy", "Branches_of_philosophy", "Philosophical_schools_and_traditions",
                  "Philosophical_movements", "Philosophical_concepts", "Philosophical_theories",
                  "Philosophical_arguments", "Philosophers", "Philosophical_literature", 
                  "History_of_philosophy", "Philosophy_by_period", "Philosophy_by_region",
                  "Aesthetics", "Epistemology", "Ethics", "Logic", "Metaphysics", "Social_philosophy",
                  "Thought", "Attention", "Cognition", "Cognitive_biases", "Creativity",
                  "Decision_theory", "Emotion", "Error", "Imagination", "Intelligence", "Learning",
                  "Memory", "Strategic_management", "Perception", "Problem_solving",
                  "Psychological_adjustment", "Psychometrics"]
    elif main_cate == 'culture':
        cates = ["Culture", "Humanities", "Classical_studies", "Critical_theory", "Cultural_anthropology",
                  "Folklore", "Food_and_drink", "Literature", "Museology", "Mythology", "Popular_culture",
                  "Science_and_culture", "Traditions", "The_arts", "Entertainment", "Celebrity", 
                  "Censorship_in_the_arts", "Festivals", "Parties", "Poetry", "Performing_arts", "Visual_arts",
                  "Architecture", "Comics", "Crafts", "Design", "Drawing", "Film", "New_media_art", "Painting",
                  "Photography", "Sculpture", "Storytelling", "Theatre", "Games", "Toys", "Board_games",
                  "Card_games", "Dolls", "Puppetry", "Puzzles", "Role-playing_games", "Video_games",
                  "Sports", "Recreation", "Air_sports", "American_football", "Association_football",
                  "Auto_racing", "Baseball", "Basketball", "Boating", "Boxing", "Canoeing", "Cricket", "Cycling",
                  "Physical_exercise", "Fishing", "Golf", "Gymnastics", "Hobbies", "Horse_racing", "Ice_hockey",
                  "Lacrosse", "Olympic_Games", "Rugby_league", "Rugby_union", "Sailing", "Skiing", "Swimming",
                  "Tennis", "Track_and_field", "Walking", "Water_sports", "Whitewater_sports", "Mass_media",
                  "Broadcasting", "Internet", "Magazines", "Newspapers", "Publications",
                  "Publishing", "Television", "Radio"]
    elif main_cate == 'geography':
        cates = ["Geography", "Places", "Earth", "World", "Bodies_of_water", "Cities", "Communities",
                  "Continents", "Countries", "Deserts", "Lakes", "Landforms", "Mountains", "Navigation",
                  "Oceans", "Populated_places", "Protected_areas", "Regions", "Rivers", "Subterranea_(geography)",
                  "Territories", "Towns", "Villages"]
    elif main_cate == 'health':
        cates = ["Health_promotion", "Life_extension", "Prevention", "Sexual_health", "Sleep", "Beauty",
                  "Nutrition", "Dietary_supplements", "Dietetics", "Nutrients", "Amino_acids", "Dietary_minerals",
                  "Nootropics", "Phytochemicals", "Vitamins", "Nutritional_advice_pyramids", "Physical_exercise",
                  "Bodyweight_exercise", "Cycling", "Exercise_equipment", "Exercise_instructors", "Dance",
                  "Exercise_physiology", "Hiking", "Pilates", "Running", "Sports", "Swimming", "Tai_chi", "Walking",
                  "Weight_training", "Yoga", "Hygiene", "Cleaning", "Oral_hygiene", "Positive_psychology",
                  "Mental_health", "Psychotherapy", "Public_health", "Health_by_country", "Health_law", "Health_promotion",
                  "Health_standards", "Hospitals", "Occupational_safety_and_health", "Pharmaceutical_industry",
                  "Pharmaceuticals_policy", "Safety", "Health_sciences", "Clinical_research", "Diseases_and_disorders",
                  "Epidemiology", "Midwifery", "Nursing", "Optometry", "Pharmacy", "Public_health", "Medicine",
                  "Alternative_medicine", "Cardiology", "Endocrinology", "Forensic_science", "Gastroenterology", "Genetics",
                  "Geriatrics", "Gerontology", "Hematology", "Nephrology", "Neurology", "Obstetrics", "Oncology",
                  "Ophthalmology", "Orthopedic_surgical_procedures", "Pathology", "Pediatrics", "Psychiatry", 
                  "Rheumatology", "Surgery", "Urology", "Dentistry", "Oral_hygiene", "Orthodontics", "Veterinary_medicine"]
    elif main_cate == 'history':
        cates = ["History", "Historiography", "History_of_Africa", "History_of_Asia", "History_of_Europe", 
                  "History_of_the_Americas", "History_of_North_America", "History_of_South_America", "History_of_Central_Europe",
                  "History_of_the_Middle_East", "History_of_Oceania", "Empires", "History_by_city"]
        cates += parse_subcate_list(start, end, gcmtitle="Category:History_by_country")
    elif main_cate == 'human':
        cates = ["Human_impact_on_the_environment", "Climate_change", "Nature_conservation", "Deforestation", 
                  "Environmentalism", "Pollution", "Human_overpopulation", "Urbanization"]
        cates += parse_subcate_list(start, end, gcmtitle="Category:Human_activities")
    elif main_cate == 'nature':
        cates = ["Biology", "Botany", "Ecology", "Health_sciences", "Medicine", "Neuroscience", "Zoology",
                  "Earth_sciences", "Atmospheric_sciences", "Geography", "Geology", "Geophysics",
                  "Oceanography", "Nature", "Animals", "Natural_environment", "Humans", "Life",
                  "Natural_resources", "Plants", "Physical_sciences", "Astronomy", "Chemistry",
                  "Climate", "Physics", "Space", "Universe", "Scientific_method", "Scientists"]
        cates += parse_subcate_list(start, end, gcmtitle="Category:Natural_resources")
        cates += parse_subcate_list(start, end, gcmtitle="Category:Scientists")
    elif main_cate == 'people':
        cates = ["People", "Beginners_and_newcomers", "Children", "Heads_of_state", "People_by_legal_status",
                  "Men", "Old_age", "Political_people", "Rivalry", "Social_groups", "Subcultures", "Women",
                  "Personal_timelines", "Terms_for_men", "Activists", "Actors", "Astronauts", "Billionaires",
                  "Chief_executive_officers", "Composers", "Cyborgs", "Defectors", "Generals", "Humanitarians",
                  "Innovators", "Inventors", "LGBT_people", "Monarchs", "Musicians", "Musical_groups", 
                  "Photographers", "Politicians", "Presidents", "Princes", "Princesses", "Revolutionaries",
                  "Settlers", "Singers", "Slaves", "Victims", "People_associated_with_war", "Writers",
                  "Self", "Alter_egos", "Consciousness_studies", "Gender", "Personality", "Human_sexuality",
                  "Sexual_orientation", "Personal_life", "Clothing", "Employment", "Food_and_drink",
                  "Games", "Health", "Home", "Income", "Interpersonal_relationships", "Leisure", "Love",
                  "Motivation", "Personal_development", "Pets"]
        subcates = ["People_by_city", "People_by_company", "People_by_continent", "People_by_educational_institution", "People_by_ethnicity", 
                    "People_by_nationality", "People_by_medical_or_psychological_condition", "People_by_occupation",
                    "People_by_political_orientation", "People_by_religion", "People_by_revolution", "People_by_status", "Personal_life"]
        for subcate in subcates:
            cates += parse_subcate_list(start, end, gcmtitle=f"Category:{subcate}")
    else:
        cates = []
        print(f'Unrecognized main category: {main_cate}!!!')
    return cates

def get_news_cates_list(main_cate):
    cates = ['Published', 'Original_reporting']
    return cates

def dump_revision_to_json(start, end, cates, tmp_path):
    for cate in cates:
        gcmcontinue = ''
        gcmtitle = f"Category:{cate}"
        print(gcmtitle)
        with open(f'{tmp_path}/raw_revisions_{gcmtitle}.json', 'a') as json_file:
            pages, gcmcontinue = get_data(start, end, gcmcontinue, gcmtitle)
            out = parse_revision(pages, gcmcontinue, json_file)
            print(f'Write {len(out)} data into raw_revisions_{gcmtitle}.json')
            while len(gcmcontinue)>0:
                pages, gcmcontinue = get_data(start, end, gcmcontinue, gcmtitle)
                out = parse_revision(pages, gcmcontinue, json_file)
                print(f'Write {len(out)} data into raw_revisions_{gcmtitle}.json')
            
                
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default='news', type=str,
                        help="Specify document domain, please select from: wiki, news")
    # Available categories in wiki: philosophy, culture, geography, health, history, human, nature, people
    # Available categories in news: all
    parser.add_argument('--main_cate', default='all', type=str,
                        help="Specify category of documents under each domain")
    args = parser.parse_args()
    
    
    start, end = get_time_range()
    S = requests.Session()
    cates = []
    if args.domain == 'wiki':
        tmp_path = f'../data/{args.domain}'
        tmp_file_path = f'../data/{args.domain}/raw'
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
        if not os.path.isdir(tmp_file_path):
            os.mkdir(tmp_file_path)
        URL = "https://en.wikipedia.org/w/api.php"
        cates = get_wiki_cates_list(args.main_cate)
    else:
        tmp_path = f'../data/{args.domain}'
        tmp_file_path = f'../data/{args.domain}/raw'
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
        if not os.path.isdir(tmp_file_path):
            os.mkdir(tmp_file_path)
        URL = "https://www.wikinews.org/w/api.php"
        cates = get_news_cates_list(args.main_cate)
    
    tmp_path = f'../data/{args.domain}/raw/{args.main_cate}'
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    
    dump_revision_to_json(start, end, cates, tmp_path)

