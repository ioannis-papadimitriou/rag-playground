import requests
import arxiv
import os
import time
import logging
from datetime import datetime
import hashlib
import sqlite3
from pathlib import Path
import json
from urllib.parse import quote

class ArticleDownloader:
    def __init__(self, output_dir="qa_docs", 
                 relevant_keywords=None,
                 irrelevant_keywords=None,
                 relevant_categories=None,
                 subject_specific_filters=None):
        """
        Initialize the downloader with customizable filters.
        """
        self.output_dir = output_dir
        self.relevant_keywords = relevant_keywords or []
        self.irrelevant_keywords = irrelevant_keywords or []
        self.relevant_categories = relevant_categories or []
        self.subject_specific_filters = subject_specific_filters or {}
        
        self.setup_logging()
        self.create_output_directory()
        self.setup_database()

    def setup_logging(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f'article_downloader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)

    def setup_database(self):
        self.db_path = os.path.join(self.output_dir, "articles.db")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    title TEXT,
                    url TEXT,
                    file_hash TEXT PRIMARY KEY,
                    filename TEXT,
                    download_date TIMESTAMP,
                    subject TEXT,
                    source TEXT,
                    doi TEXT,
                    authors TEXT,
                    category TEXT
                )
            """)

    def create_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory: {self.output_dir}")

    def construct_query(self, subject):
        """Construct search query using subject-specific filters."""
        if subject in self.subject_specific_filters:
            return self.subject_specific_filters[subject]
        return subject

    def is_relevant_article(self, result, subject):
        """Check if article is relevant based on configured filters."""
        try:
            title = result.title.lower()
            abstract = result.summary.lower()
            category = result.primary_category.lower()

            # Check for irrelevant keywords
            if any(keyword.lower() in title or keyword.lower() in abstract 
                  for keyword in self.irrelevant_keywords):
                logging.info(f"Filtered out due to irrelevant keywords: {result.title}")
                return False

            # Check for relevant keywords
            if self.relevant_keywords:
                if not any(keyword.lower() in title or keyword.lower() in abstract 
                          for keyword in self.relevant_keywords):
                    logging.info(f"Filtered out due to missing relevant keywords: {result.title}")
                    return False

            # Check categories
            if self.relevant_categories:
                if not any(cat.lower() in category for cat in self.relevant_categories):
                    logging.info(f"Filtered out due to irrelevant category: {result.title}")
                    return False

            return True

        except Exception as e:
            logging.error(f"Error checking relevance: {str(e)}")
            return True

    def search_arxiv(self, query, max_results=None):
        try:
            enhanced_query = self.construct_query(query)
            client = arxiv.Client()
            
            search = arxiv.Search(
                query=enhanced_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            logging.info(f"Searching arXiv for: {enhanced_query}")
            articles_found = 0
            articles_downloaded = 0

            for result in client.results(search):
                if self.is_relevant_article(result, query):
                    articles_found += 1
                    logging.info(f"Found relevant article {articles_found}: {result.title}")
                    
                    try:
                        safe_title = "".join(x for x in result.title if x.isalnum() or x in (' ', '-', '_'))
                        filename = f"{safe_title[:100]}.pdf"
                        filepath = os.path.join(self.output_dir, filename)

                        if os.path.exists(filepath):
                            logging.info(f"Article already exists: {filename}")
                            continue

                        logging.info(f"Downloading: {filename}")
                        result.download_pdf(dirpath=self.output_dir, filename=filename)
                        
                        if not os.path.exists(filepath):
                            logging.error(f"Download failed for: {filename}")
                            continue
                            
                        self.record_download(
                            title=result.title,
                            url=result.pdf_url,
                            filepath=filepath,
                            subject=query,
                            source="arxiv",
                            doi=result.doi,
                            authors=", ".join([author.name for author in result.authors]),
                            category=result.primary_category
                        )
                        articles_downloaded += 1
                        logging.info(f"Successfully downloaded article {articles_downloaded}")
                        
                        time.sleep(1)

                    except Exception as e:
                        logging.error(f"Error downloading article {result.title}: {str(e)}")
                        if os.path.exists(filepath):
                            try:
                                os.remove(filepath)
                            except:
                                pass
                        continue

            logging.info(f"Completed arXiv search. Found {articles_found}, Downloaded {articles_downloaded} relevant articles")
            return articles_downloaded

        except Exception as e:
            logging.error(f"Error in arXiv search: {str(e)}")
            return 0

    def record_download(self, title, url, filepath, subject, source, doi=None, authors=None, category=None):
        try:
            file_hash = self.calculate_file_hash(filepath)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO articles 
                    (title, url, file_hash, filename, download_date, subject, source, doi, authors, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (title, url, file_hash, os.path.basename(filepath), 
                      datetime.now().isoformat(), subject, source, doi, authors, category))
        except Exception as e:
            logging.error(f"Error recording download: {str(e)}")
            raise

    def calculate_file_hash(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_download_stats(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_articles,
                        COUNT(DISTINCT subject) as unique_subjects,
                        COUNT(DISTINCT category) as unique_categories,
                        MIN(download_date) as first_download,
                        MAX(download_date) as last_download
                    FROM articles
                """)
                stats = cursor.fetchone()
                
                cursor.execute("""
                    SELECT subject, COUNT(*) as count 
                    FROM articles 
                    GROUP BY subject 
                    ORDER BY count DESC
                """)
                subject_counts = cursor.fetchall()
                
                cursor.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM articles 
                    GROUP BY category 
                    ORDER BY count DESC
                """)
                category_counts = cursor.fetchall()
                
                return {
                    "total_articles": stats[0],
                    "unique_subjects": stats[1],
                    "unique_categories": stats[2],
                    "first_download": stats[3],
                    "last_download": stats[4],
                    "articles_by_subject": dict(subject_counts),
                    "articles_by_category": dict(category_counts)
                }
        except Exception as e:
            logging.error(f"Error getting stats: {str(e)}")
            return {
                "total_articles": 0,
                "unique_subjects": 0,
                "unique_categories": 0,
                "articles_by_subject": {},
                "articles_by_category": {}
            }

if __name__ == "__main__":
    # Define your filters and configurations here
    RELEVANT_KEYWORDS = [
        "materials", "material properties", "material characterization",
        "material synthesis", "materials engineering", "metallurgy",
        "ceramics", "polymers", "composites", "semiconductors",
        "crystal structure", "microstructure", "mechanical properties",
        "thermal properties", "electronic properties", "optical properties",
        "nanomaterials", "biomaterials", "materials processing"
    ]

    IRRELEVANT_KEYWORDS = [
        "movie", "film", "cinema", "theatre", "psychology", 
        "marketing", "advertisement", "social media", "economics",
        "literature", "music", "art history", "philosophy"
    ]

    RELEVANT_CATEGORIES = [
        "cond-mat", "physics", "chemistry", "physics.app-ph",
        "physics.mtrl-sci", "cond-mat.mtrl-sci", "cond-mat.soft"
    ]

    # Define subject-specific search queries
    SUBJECT_FILTERS = {
        "materials science": 'ti:"materials science" OR abs:"materials science" OR "material properties" OR "materials engineering"',
        "materials science and engineering": 'ti:"materials engineering" OR abs:"materials engineering" OR "materials processing"',
        "materials": 'ti:"materials" AND "properties" OR ti:"materials science"'
    }

    # Create downloader instance with configurations
    downloader = ArticleDownloader(
        output_dir="materials_papers",
        relevant_keywords=RELEVANT_KEYWORDS,
        irrelevant_keywords=IRRELEVANT_KEYWORDS,
        relevant_categories=RELEVANT_CATEGORIES,
        subject_specific_filters=SUBJECT_FILTERS
    )

    # Define subjects to search
    subjects = [
        "materials science",
        "materials science and engineering",
        "materials"
    ]

    # Download articles
    total_downloaded = 0
    for subject in subjects:
        print(f"\nSearching for {subject} articles...")
        downloaded = downloader.search_arxiv(subject, max_results=50)
        total_downloaded += downloaded
        print(f"Downloaded {downloaded} articles for {subject}")

    # Print final statistics
    stats = downloader.get_download_stats()
    print("\nDownload Statistics:")
    print(f"Total articles: {stats['total_articles']}")
    print("\nArticles by subject:")
    for subject, count in stats['articles_by_subject'].items():
        print(f"- {subject}: {count}")
    print("\nArticles by category:")
    for category, count in stats['articles_by_category'].items():
        print(f"- {category}: {count}")