import requests
import time


class DiscourseScraper:
    def __init__(self, hostname):
        self.hostname = hostname
        self.topic_ids = []
        self.request_count = 0
        self.start_time = time.time()

    def make_request(self, url, params=None):
        # Ensure 200 requests per minute rate limit
        # 60 seconds divided by 200 requests equals 0.3 seconds between requests
        if self.request_count >= 200:
            elapsed_time = time.time() - self.start_time

            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

            # Reset the request count and timer after waiting
            self.request_count = 0
            self.start_time = time.time()

        response = requests.get(url, params=params)
        self.request_count += 1

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed: {response.status_code}")
            return None

    def get_topics(self):
        page = 0

        while True:
            url = f"https://{self.hostname}/top.json"
            params = {"period": "all", "page": page}
            content = self.make_request(url, params=params)

            if content is None:
                break

            # Extract the 'topics' list
            topics = content.get("topic_list", {}).get("topics", [])

            # If 'topics' is an empty list, we're done
            if not topics:
                break

            # Otherwise, extract the 'id' of each topic and add it to our list
            for topic in topics:
                self.topic_ids.append(topic.get("id"))

            # Increment the page number for the next iteration
            page += 1

        return self.topic_ids

    def fetch_topic_and_posts(self, topic_id):
        topic_url = f"https://{self.hostname}/t/{topic_id}.json"
        topic_data = self.make_request(topic_url)

        if topic_data is None:
            return None

        stream_ids = set(topic_data.get('post_stream', {}).get('stream', []))
        post_ids = set(post['id'] for post in topic_data.get('post_stream', {}).get('posts', []))

        missing_ids = stream_ids - post_ids

        if missing_ids:

            post_url = f"https://{self.hostname}/t/{topic_id}/posts.json"
            post_data = self.make_request(post_url, params={'post_ids[]': list(missing_ids)})

            if post_data:
                returned_post_ids = [post["id"] for post in post_data.get("post_stream",{}).get('posts',[])]
                #in the order of the original topic
                for id in topic_data.get('post_stream', {}).get('stream', []):
                    #if the post is missing, but contained in the response
                    if id in missing_ids and id in returned_post_ids:
                        post = next(filter(lambda x: x["id"] == id,post_data["post_stream"]["posts"]))
                        topic_data["post_stream"]["posts"].append(post)


        return topic_data  # Contains topic info along with all posts, including those initially missing
