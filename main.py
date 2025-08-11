import requests
import json

# Verified institution IDs
INSTITUTIONS = {
    "University_of_Toronto": "I185261750",
    "McMaster_University": "I98251732",
    "University_of_Waterloo": "I151746483",
    "Queens_University": "I204722609",
    "University_of_Guelph": "I79817857"
}

HEADERS = {
    "User-Agent": "CiteNova/1.0 (mailto:2018dgscmt@gmail.com)"
}


#open alex api saves abstracts of publications through an inversed index
#words have a specific location (eg. machine will be saved as [5])
def decode_abstract(inv_idx):

    #if the abstract of publication is empty, return nothing
    if not inv_idx:
        return None

    #empty list to store positions
    positions = []

    #loops over each words and its position then appends to positions
    for token, idxs in inv_idx.items():
        for i in idxs:
            positions.append((i, token))

    #sorts the list so that the words are in reading order 
    positions.sort(key=lambda x: x[0])

    #forms the sentence of whatever the abstract is
    words = [t for _, t in positions]
    text = " ".join(words)
    text = (text.replace(" ,", ",").replace(" .", ".").replace(" ;", ";")
                .replace(" :", ":").replace(" !", "!").replace(" ?", "?")
                .replace(" '", "'").replace("( ", "(").replace(" )", ")"))
    return text


#fetches the data from open alex
def fetch_recent(inst_id):
    url = "https://api.openalex.org/works"

    #filters through the institution id (each university has a specific code)
    params = {
        "filter": f"institutions.id:https://openalex.org/{inst_id}",
        "sort": "publication_date:desc",

        #finds the most 100 recent publications from each institution
        "per-page": 100,
        "select": "id,title,doi,abstract_inverted_index"
    }
    r = requests.get(url, params=params, headers=HEADERS)
    r.raise_for_status()
    return r.json()["results"]

if __name__ == "__main__":
    for uni_name, inst_id in INSTITUTIONS.items():
        papers = fetch_recent(inst_id)

        #stores data about the publication (doi --> link to the publication, abstract (summary/overview of publication))
        minimal_data = [{
            "id": p["id"],
            "title": p.get("title"),
            "doi": p.get("doi"),
            "abstract": decode_abstract(p.get("abstract_inverted_index"))
        } for p in papers]

        #saves as JSON file for said university
        file_name = f"{uni_name}.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(minimal_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(minimal_data)} records to {file_name}")
