import random
import xmlrpc.client
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
import os

# Config
url = os.getenv('SITE_URL') + '/xmlrpc.php'
username = os.getenv('WP_USER')
password = os.getenv('WP_PASS')

client = Client(url, username, password)

titles = [
    "Top 10 Job Interview Tips for Fresh Graduates",
    "How to Write a Winning CV in 2025",
    "Remote Work Trends in Mauritius",
    "Best IT Jobs in Mauritius Right Now",
    "How to Start Freelancing from Home",
    "Salary Guide: What Jobs Pay in Mauritius",
    "Career Change at 30: Is It Too Late?",
    "How to Network Like a Pro in Mauritius",
    "Top 5 In-Demand Skills for 2025",
    "How to Get Promoted in 6 Months"
]

def generate_paragraphs(n=5):
    words = ["career", "job", "skills", "interview", "resume", "Mauritius", "work", "salary", "opportunity", "growth", "success", "professional", "experience", "employer", "candidate"]
    para = ""
    for _ in range(n):
        para += " ".join(random.choices(words, k=80)) + ". "
    return para.strip()

for i, title in enumerate(titles):
    post = WordPressPost()
    post.title = title
    post.content = f"<h2>{title}</h2>" + generate_paragraphs(6)
    post.terms_names = {'category': ['Jobs', 'Career']}
    post.post_status = 'publish'
    client.call(NewPost(post))
    print(f"Created: {title}")
