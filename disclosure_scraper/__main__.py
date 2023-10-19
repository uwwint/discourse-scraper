from . import DiscourseScraper
import json

if __name__ == "__main__":
    d = DiscourseScraper("help.galaxyproject.org")
    for topic_id in d.get_topics():
        topic = d.fetch_topic_and_posts(topic_id)
        json.dump(topic, open(f"data/{topic_id}.json","w"))