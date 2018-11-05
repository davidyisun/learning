# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
from scrapy.exceptions import DropItem
import codecs


class CnblogspiderPipeline(object):
    def __init__(self):
        self.file = codecs.open('papers.json', 'w', 'utf-8')
    def process_item(self, item, spider):
        self.file.write('---'*20+'\n')
        if item['title']:
            # line = json.dumps(dict(item)) + '\n'
            # self.file.write(line)
            self.file.write(item['title'] + '\n')
            return item
        else:
            raise DropItem("Missing title in %s" % item)
