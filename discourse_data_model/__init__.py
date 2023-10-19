import json
from typing import List, Optional, Any, Tuple, Dict
from dataclasses import dataclass, field, fields, is_dataclass
import os
from datetime import datetime
import pandas as pd
import json
import re
from html import unescape


@dataclass
class User:
    id: int
    username: str
    name: str
    avatar_template: str


@dataclass
class LinkCount:
    url: str
    internal: bool
    reflection: bool
    title: str
    clicks: int


@dataclass
class ActionSummary:
    id: int
    count: int
    hidden: Optional[bool] = None
    can_act: Optional[bool] = None


@dataclass
class Post:
    id: int
    name: str
    username: str
    avatar_template: str
    created_at: str
    cooked: str
    post_number: int
    post_type: int
    updated_at: str
    reply_count: int
    reply_to_post_number: Optional[int]
    quote_count: int
    incoming_link_count: int
    reads: int
    readers_count: int
    score: float
    yours: bool
    topic_id: int
    topic_slug: str
    display_username: str
    primary_group_name: Optional[str]
    flair_name: Optional[str]
    flair_url: Optional[str]
    flair_bg_color: Optional[str]
    flair_color: Optional[str]
    flair_group_id: Optional[int]
    version: int
    can_edit: bool
    can_delete: bool
    can_recover: bool
    can_see_hidden_post: bool
    can_wiki: bool
    read: bool
    user_title: Optional[str]
    bookmarked: bool
    actions_summary: List[ActionSummary]
    moderator: bool
    admin: bool
    staff: bool
    user_id: int
    hidden: bool
    trust_level: int
    deleted_at: Optional[str]
    user_deleted: bool
    edit_reason: Optional[str]
    can_view_edit_history: bool
    wiki: bool
    can_accept_answer: bool
    can_unaccept_answer: bool
    accepted_answer: bool
    topic_accepted_answer: bool
    link_counts: Optional[List[LinkCount]] = None
    can_vote: Optional[bool] = None
    reply_to_user: Optional[User] = None
    via_email: Optional[bool] = None
    action_code: Optional[str] = None
    polls: Optional[List] = None


@dataclass
class PostStream:
    posts: List[Post]
    stream: List[int]


@dataclass
class Poster:
    extras: Optional[str]
    description: str
    user: User


@dataclass
class SuggestedTopic:
    id: int
    title: str
    fancy_title: str
    slug: str
    posts_count: int
    reply_count: int
    highest_post_number: int
    image_url: Optional[str]
    created_at: str
    last_posted_at: str
    bumped: bool
    bumped_at: str
    archetype: str
    unseen: bool
    pinned: bool
    unpinned: Optional[bool]
    visible: bool
    closed: bool
    archived: bool
    bookmarked: Optional[bool]
    liked: Optional[bool]
    tags: List[str]
    tags_descriptions: dict
    like_count: int
    views: int
    category_id: int
    featured_link: Optional[str]
    has_accepted_answer: bool
    posters: List[Poster]


@dataclass
class Thumbnail:
    max_width: Optional[int]
    max_height: Optional[int]
    width: int
    height: int
    url: str


@dataclass
class AcceptedAnswer:
    post_number: int
    username: str
    excerpt: str
    name: Optional[str]


@dataclass
class Participant:
    id: int
    username: str
    name: str
    avatar_template: str
    post_count: int
    primary_group_name: Optional[str]
    flair_name: Optional[str]
    flair_url: Optional[str]
    flair_color: Optional[str]
    flair_bg_color: Optional[str]
    flair_group_id: Optional[int]
    trust_level: int
    admin: Optional[bool] = None
    moderator: Optional[bool] = None


@dataclass
class Link:
    url: str
    title: str
    internal: bool
    attachment: bool
    reflection: bool
    clicks: int
    user_id: int
    domain: str
    root_domain: str


@dataclass
class Details:
    can_edit: bool
    notification_level: int
    participants: List[Participant]
    created_by: User
    last_poster: User
    links: List[Link]


@dataclass
class TopicData:
    post_stream: PostStream
    timeline_lookup: List[List[int]]
    tags: List[str]
    tags_descriptions: dict
    id: int
    title: str
    fancy_title: str
    posts_count: int
    created_at: str
    views: int
    reply_count: int
    like_count: int
    last_posted_at: str
    visible: bool
    closed: bool
    archived: bool
    has_summary: bool
    archetype: str
    slug: str
    category_id: int
    word_count: int
    deleted_at: Optional[str]
    user_id: int
    featured_link: Optional[str]
    pinned_globally: bool
    pinned_at: Optional[str]
    pinned_until: Optional[str]
    image_url: Optional[str]
    slow_mode_seconds: int
    draft: Optional[str]
    draft_key: str
    draft_sequence: Optional[int]
    unpinned: Optional[bool]
    pinned: bool
    current_post_number: int
    highest_post_number: int
    deleted_by: Optional[str]
    actions_summary: List[ActionSummary]
    chunk_size: int
    bookmarked: bool
    bookmarks: List[Any]
    topic_timer: Optional[Any]
    message_bus_last_id: int
    participant_count: int
    show_read_indicator: bool
    thumbnails: List[Thumbnail]
    slow_mode_enabled_until: Optional[str]
    summarizable: bool
    tags_disable_ads: bool
    can_vote: bool
    vote_count: int
    user_voted: bool
    discourse_zendesk_plugin_zendesk_id: Optional[int]
    discourse_zendesk_plugin_zendesk_url: str
    details: Details
    accepted_answer: Optional[AcceptedAnswer] = None
    suggested_topics: Optional[List[SuggestedTopic]] = field(default_factory=list)
    featured_link_root_domain: Optional[str] = None
    unicode_title: Optional[str] = None

    def get_posts_until_solution(self) -> List[Post]:
        """
        Retrieve all posts from the beginning until the post that's marked as the solution.

        Returns:
            List[Post]: A list of posts up to and including the solution post.
        """
        posts_until_solution = []

        # Check if there's an accepted answer in the topic
        if self.accepted_answer:
            # Iterate through the posts
            for post in self.post_stream.posts:
                # Add post to the list
                posts_until_solution.append(post)

                # Check if the current post is the solution
                if post.accepted_answer:
                    break

        return posts_until_solution

    @staticmethod
    def clean_html(raw_html: str) -> str:
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return unescape(cleantext)

    def posts_to_dict(self, posts: List[Post]) -> List[Dict]:
        if not posts:
            return []

        first_post_username = posts[0].username
        transformed_posts = []

        for post in posts:
            role = "user" if post.username == first_post_username else "system"
            text = self.clean_html(post.cooked)
            transformed_posts.append({"role": role, "text": text})

        return transformed_posts

    def get_solution_post_dict(self) -> List[Dict]:
        posts_before_and_including_solution = self.get_posts_until_solution()
        return self.posts_to_dict(posts_before_and_including_solution)

    @staticmethod
    def from_json(json_dict: dict) -> 'TopicData':
        json_dict['post_stream'] = PostStream(posts=[Post(**post) for post in json_dict['post_stream']['posts']],
                                              stream=json_dict['post_stream']['stream'])

        if json_dict.get('accepted_answer'):
            json_dict['accepted_answer'] = AcceptedAnswer(**json_dict['accepted_answer'])

        json_dict['actions_summary'] = [ActionSummary(**item) for item in json_dict['actions_summary']]

        json_dict['details'] = Details(can_edit=json_dict['details']['can_edit'],
                                       notification_level=json_dict['details']['notification_level'],
                                       participants=[Participant(**participant) for participant in
                                                     json_dict['details']['participants']],
                                       created_by=User(**json_dict['details']['created_by']),
                                       last_poster=User(**json_dict['details']['last_poster']),
                                       links=[] if "links" not in json_dict["details"] else [Link(**link) for link in json_dict['details']['links']])
        return TopicData(**json_dict)


if __name__ == "__main__":
    def load_json_file(file_path: str) -> TopicData:
        with open(file_path, 'r', encoding='utf-8') as file:
            return TopicData.from_json(json.load(file))


    def load_and_parse_json_files_from_directory(directory_path: str) -> List['TopicData']:
        topic_data_objects = []

        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                topic_data = load_json_file(file_path)
                topic_data_objects.append((filename,topic_data))

        return topic_data_objects

    def is_primitive(field_type) -> bool:
        """Check if a type is a primitive type."""
        # Modify this set to include/exclude types you consider primitive
        primitive_types = {int, str, bool, float, type(None)}
        return field_type in primitive_types


    def extract_primitive_fields(topic_data: TopicData, filename: str) -> dict:
        """Extract the primitive fields from a TopicData instance."""
        primitive_data = {'filename': filename}

        for field in fields(topic_data):
            # Check if the field type is primitive or not
            if is_primitive(field.type):
                primitive_data[field.name] = getattr(topic_data, field.name)

            if field.name == 'accepted_answer':
                primitive_data["accepted_answer"] = getattr(topic_data, field.name) != None


        return primitive_data


    def create_dataframe_from_topicdata(file_topicdata_pairs: List[Tuple[str, TopicData]]) -> pd.DataFrame:
        """Create a DataFrame from a list of (filename, TopicData) pairs."""
        # Extract primitive fields from each TopicData instance
        data_for_dataframe = [
            extract_primitive_fields(topic_data, filename)
            for filename, topic_data in file_topicdata_pairs
        ]

        # Create a DataFrame
        df = pd.DataFrame(data_for_dataframe)

        return df

    # Usage
    directory_path = '../data/'  # replace with your directory path
    topics = load_and_parse_json_files_from_directory(directory_path)
    df = create_dataframe_from_topicdata(topics)
    df_accepted = df[df['accepted_answer']]

    # Then, filter the DataFrame
    df_top_views = df_accepted[df_accepted['views'] >= 100]

    # Step 3: Sort by 'post_count' in descending order
    df_sorted = df_top_views.sort_values(by='posts_count', ascending=False)

    # Initialize an empty list to store the JSON strings
    posts_json_list = []

    # Iterate over each row in the DataFrame
    for index, row in df_sorted.iterrows():
        # Parse the file into a TopicData object
        topic_data = load_json_file(f"{directory_path}{row['filename']}")

        # Generate the post JSON and append it to the list
        sol_post = topic_data.get_solution_post_dict()
        posts_json_list.append(sol_post)
