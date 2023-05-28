# Mini search engine

_inspired by [seroost](https://github.com/tsoding/seroost)_

Uses [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to search for the most relevant words on a page

## How to use

First you need to index the pages in the database

You need to pass an initial link (seed)

```
python main.py index https://en.wikipedia.org/wiki/Main_Page
```

It will index all the text on that page in the database and look for links within that page, repeating the same process for each link

**To prevent the program from causing any kind of spam**, you must pass a delay for each page and the page number you want to index
```
python main.py index https://en.wikipedia.org/wiki/Main_Page -d 10 -n 100
```

It will wait 10 seconds before indexing each page and will index a maximum of 100 pages

---

After indexing you can search for something
```
python main.py search day of the week
```