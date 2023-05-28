from pathlib import Path
from subprocess import run

from main import Page

idx = Path('index.db')
link = 'https://en.wikipedia.org/wiki/Tf%E2%80%93idf'
term = 'TFIDF term document frequency'

if idx.exists():
    idx.unlink()


run(['python', 'main.py', 'index', link, '-n', '2'], check=True)

# Test index
pages = Page.select()
crawled = [page.link for page in pages if page.crawled]
non_crawled = [page.link for page in pages if not page.crawled]

assert len(crawled) == 2, 'not enough crawled pages'
assert len(non_crawled) > 2, 'not enough uncrawled pages'

# Test search
assert (
    link
    in run(
        ['python', 'main.py', 'search', link, *term.split()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
), 'link does not found in search'

print('✅ all tests passed')
