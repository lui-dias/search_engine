from collections import Counter
from dataclasses import dataclass
from math import log
from subprocess import run
from time import sleep
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import click as c
import regex as re
from alive_progress import alive_it
from nltk.stem import SnowballStemmer
from peewee import (
    BooleanField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)
from requests import Session
from selectolax.parser import HTMLParser

try:
    from rich import print
except ImportError:
    ...

session = Session()
user_agent = 'ellandore'
ss = SnowballStemmer('english')

punctuation = '[!"#$%&\'()*+,\-.\/:;<=>?@[\]^_`{|}~]'
non_punctuation = '[^!"#$%&\'()*+,\-.\/:;<=>?@[\]^_`{|}~]'


def clear():
    run('cls||clear', shell=True)


@dataclass
class Metadata:
    title: str
    description: str
    keywords: str
    author: str
    language: str


db = SqliteDatabase('index.db')


class BaseModel(Model):
    class Meta:
        database = db


class Page(BaseModel):
    link = TextField()
    total = IntegerField()
    crawled = BooleanField(default=False)


class Frequency(BaseModel):
    page = ForeignKeyField(Page, backref='frequencies')
    term = TextField()
    count = IntegerField()


class Parser:
    def __init__(self, seed, delay: int = 0):
        self.robots = RobotFileParser(seed)
        self.robots.read()

        self.seed = seed
        self.delay = delay

    def _sanitize_link(self, link):
        return re.sub(r'^//', '/', link)

    def _is_link_alive(self, link):
        return session.head(link).status_code < 400

    def _parse_html(self, link):
        return HTMLParser(session.get(link).text)

    def get_links(self, link: str):
        sleep(self.delay)

        crawled_links = set()
        html = self._parse_html(link)

        for i in html.css('a'):
            if 'href' not in i.attrs:
                continue

            link = self._sanitize_link(i.attrs['href'])

            if not self.robots.can_fetch(user_agent, link) or not self._is_link_alive(
                link
            ):
                continue

            if link.startswith('/'):
                up = urlparse(self.seed)

                link = f'{up.scheme}://{up.netloc}{link}'
            elif link.startswith('#'):
                continue

            crawled_links.add(link)

        return list(crawled_links)

    def get_metadata(self, link: str):
        html = self._parse_html(link)

        title = html.css_first('head > title')
        description = html.css_first('meta[name="description"]')
        keywords = html.css_first('meta[name="keywords"]')
        author = html.css_first('meta[name="author"]')
        language = html.css_first('html').attributes['lang']

        return Metadata(
            title=title.text() if title else None,
            description=description.attrs['content'] if description else None,
            keywords=keywords.attrs['content'] if keywords else None,
            author=author.attrs['content'] if author else None,
            language=language or 'en',
        )

    def get_text(self, link: str):
        sleep(self.delay)
        html = self._parse_html(link)

        all = html.css('*')
        just_text = [
            i
            for i in all
            if i.tag
            not in (
                'script',
                'style',
                'noscript',
                'head',
                'html',
                'body',
                'video',
                'audio',
                'iframe',
            )
        ]

        text = ' '.join([i.text(deep=False) for i in just_text]).strip()
        text = re.sub(r'\n|\r\n|\t', ' ', text)
        text = re.sub(r' {2,}', ' ', text)

        return text

    def parse_frequency(self, terms: list[str]):
        return Counter(terms)


# A lexer from scratch because nltk.tokenize.word_tokenize
# need download a tokenizer (punkt) to works
class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def peek(self):
        if self.pos < len(self.text):
            return self.text[self.pos]

    def advance(self):
        self.pos += 1

    def get_while(self, *patterns: str):
        s = ''
        while True:
            p = self.peek()

            if p is None or not all(re.search(pattern, p) for pattern in patterns):
                break

            s += self.peek()
            self.advance()
        return s

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            c = self.peek()

            if c is None:
                raise StopIteration

            if re.search(r'\s', c):
                self.get_while(r'\s')

            elif re.search(punctuation, c):
                self.get_while(punctuation)

            elif re.search(r'\w', c):
                s = self.get_while(r'\w')
                return ss.stem(s)

            else:
                # special characters from other languages
                s = self.get_while(r'[^\s\w]', non_punctuation)
                return ss.stem(s)


def tf(term_count, link_total):
    if term_count + link_total == 0:
        return 0

    return term_count / link_total


def idf(pages_count, total_term_occurence):
    n = pages_count
    m = max(total_term_occurence, 1)

    return log(n / m)


@c.command()
@c.argument('seed')
@c.option('-n', '--number', type=int, default=10, help='Number of links to fetch')
@c.option('-d', '--delay', type=int, default=0, help='Delay between requests')
@c.option('-r', '--reset', is_flag=True, help='Reset database')
def index(seed: str, number: int, delay: int, reset: bool):
    if reset:
        db.drop_tables([Page, Frequency])
        db.create_tables([Page, Frequency])

    p = Parser(seed, delay)

    non_crawled_pages = Page.select().where(~Page.crawled).limit(number)

    if len(non_crawled_pages) < number:
        n = len(non_crawled_pages)

        while n < number:
            for link in p.get_links(p.seed):
                if not Page.select().where(Page.link == link).count():
                    Page(link=link, total=0).save()

                n += 1

    for link in alive_it(Page.select(Page.link).where(~Page.crawled).limit(number)):
        text = p.get_text(link.link)

        lexer = Lexer(text)
        terms = set(lexer)

        freq = p.parse_frequency(terms)
        page = (
            Page.update(total=sum(freq.values()), crawled=True)
            .where(Page.link == link.link)
            .execute()
        )

        for term, f in freq.items():
            Frequency(page=page, term=term, count=f).save()


@c.command()
@c.argument('search', nargs=-1)
@c.option('-a', '--all', is_flag=True, help="Show all results")
def search(search: tuple[str], all: bool):
    search = ' '.join(search)

    lexer = Lexer(search)
    terms = set(lexer)

    r = {}

    pages = Page.select(Page.link, Page.total)
    pages_count = len(pages)

    for page in pages:
        total = 0

        for term in terms:
            total_term_occurence = (
                Frequency.select().where(Frequency.term == term).count()
            )

            term_count = (
                Frequency.select(Frequency.count)
                .join(Page)
                .where(Page.link == page.link, Frequency.term == term)
                .first()
            )

            term_count = term_count.count if term_count else 0

            total += tf(term_count, page.total) * idf(pages_count, total_term_occurence)

        r[page.link] = total

    sort: dict[str, int] = dict(sorted(r.items(), key=lambda x: x[1]))

    if sum((int(i != 0) for i in sort.values())) == 0:
        print('No results')
    else:
        for url, v in sort.items():
            if all:
                print(f'{v:.6f} -> {url}')
            elif v > 0:
                print(f'{v:.6f} -> {url}')


@c.group()
def cli():
    ...


cli.add_command(index)
cli.add_command(search)

if __name__ == '__main__':
    db.create_tables([Page, Frequency])
    cli()
