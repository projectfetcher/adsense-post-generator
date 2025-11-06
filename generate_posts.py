import random
import xmlrpc.client
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods.posts import NewPost
import os
import json

# Get from GitHub event payload
payload = json.loads(os.getenv('GITHUB_EVENT_PAYLOAD', '{}'))
site_url = payload.get('site_url', os.getenv('SITE_URL', ''))
wp_user = payload.get('wp_user', os.getenv('WP_USER', 'admin'))
wp_pass = payload.get('wp_pass', os.getenv('WP_PASS', ''))

if not site_url or not wp_user or not wp_pass:
    print("Missing credentials")
    exit(1)

url = site_url.rstrip('/') + '/xmlrpc.php'
client = Client(url, wp_user, wp_pass)

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
    "How to Get Promoted in 6 Months",
    "Building a Personal Brand Online",
    "Negotiating Salary in Mauritius",
    "Best Career Paths for 2025",
    "Overcoming Job Search Burnout",
    "Essential Soft Skills for Success"
]

def generate_paragraphs(n=5):
    words = ["career", "job", "skills", "interview", "resume", "Mauritius", "work", "salary", "opportunity", "growth", "success", "professional", "experience", "employer", "candidate", "development", "networking", "training", "advancement", "balance"]
    para = ""
    for _ in range(n):
        para += " ".join(random.choices(words, k=80)) + ". "
    return para.strip()

for i, title in enumerate(titles):
    post = WordPressPost()
    post.title = title
    post.content = f"<h2>{title}</h2><p>{generate_paragraphs(6)}</p><p>This post provides valuable insights for job seekers in Mauritius. Stay tuned for more career advice!</p>"
    post.terms_names = {'category': ['Jobs', 'Career']}
    post.post_status = 'publish'
    client.call(NewPost(post))
    print(f"Created: {title} ({len(post.content.split())} words)")

print("All 15 posts generated successfully!")
