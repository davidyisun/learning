#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    脚本名: neo4j 初识
Created on 2018-10-26
@author:David Yisun
@group:data
"""
def t1():
    from py2neo import Node,Relationship
    # new nodes
    a = Node("Person", name="Alice")
    b = Node("Person", name="Bob")
    c = Node("Person", name="Cissy")
    # new relationships
    ab = Relationship(a, "KNOWS", b)
    ac = Relationship(a, "KNOWS", c)
    # print
    print(ab)
    print(ac)

def t2():
    from py2neo import Graph, Node, Relationship
    # # new graph
    graph = Graph('http://127.0.0.1:7474',
                  username='neo4j',
                  password='huerge4040')
    # new nodes
    a = Node('Person', name='Alice')
    b = Node('Person', name='Bob')
    a['age'] = 20
    b['age'] = 21


    # new relationships
    ab = Relationship(a, 'KNOWS', b)
    ab['time'] = '2017/08/31'
    print(a)
    print(type(a))
    print(b)
    print(ab)
    # 设置默认属性
    a.setdefault('location', '上海')
    print(a)

    # 批量更新
    data = {
        'name': 'Amy',
        'age': 21,
        'gengder': 'male'
    }
    a.update(data)
    print(a)

    # # create object
    # graph.create(a)
    # graph.create(b)
    # graph.create(ab)
    # # find relationship via node
    # relationship_a = graph.match_one(start_node=a)
    # print(relationship_a)
    return


def t3():
    from py2neo import Node, Relationship, Graph

    a = Node('Person', name='Alice')
    b = Node('Person', name='Bob')
    r = Relationship(a, 'KNOWS', b)
    s = a | b | r
    print(s)
    print(type(s))

    graph = Graph('http://127.0.0.1:7474',
                  username='neo4j',
                  password='huerge4040')
    graph.create(s)
    return


def t4():
    from py2neo import Node, Relationship

    a = Node('Person', name='Alice')
    b = Node('Person', name='Bob')
    c = Node('Person', name='Mike')
    ab = Relationship(a, "KNOWS", b)
    ac = Relationship(a, "KNOWS", c)
    w = ab + Relationship(b, "LIKES", c) + ac
    # print(w)
    # print(type(w))
    from py2neo import walk

    for item in walk(w):
        print('>>'*30)
        print(item)
        print(type(item))


def t5():
    # 查询数据
    from py2neo import Node, Relationship, Graph
    graph = Graph('http://127.0.0.1:7474',
                  username='neo4j',
                  password='huerge4040')
    cql = 'MATCH (a:Person) RETURN a.name LIMIT 4'
    data = graph.data(cql)
    print(data)
    print(type(data))


def t6():
    # 查询数据
    from py2neo import Node, Relationship, Graph
    graph = Graph('http://127.0.0.1:7474',
                  username='neo4j',
                  password='huerge4040')
    print('find one:')
    node = graph.find_one(label='Person')
    print(node)
    print('>>>'*20)

    print('find:')
    nodes = list(graph.find(label='Person'))
    print(nodes)
    print('>>>'*20)

    print('match:')
    relationship = graph.match_one(rel_type='KNOWS')
    print(relationship)
    print('>>>'*20)

    print('match:')
    relationships = list(graph.match(rel_type='KNOWS'))
    print(relationships)


def t7():
    from py2neo import Node, Relationship, Graph, NodeSelector
    graph = Graph('http://127.0.0.1:7474',
                  username='neo4j',
                  password='huerge4040')
    a = Node('Person', name='Alice', age=21, location='广州')
    b = Node('Person', name='Bob', age=22, location='上海')
    c = Node('Person', name='Mike', age=21, location='北京')
    r1 = Relationship(a, 'KNOWS', b)
    r2 = Relationship(b, 'KNOWS', c)
    graph.create(a)
    graph.create(r1)
    graph.create(r2)

    # 通过 nodeselector进行筛选
    # ----- 1
    # selector = NodeSelector(graph)
    # persons = selector.select('Person', age=21)
    # print(persons)
    # print(list(persons))
    # ----- 2
    # selector = NodeSelector(graph)
    # persons = selector.select('Person').where('_.name =~ "A.*"')
    # print(list(persons))
    # persons = selector.select('Person').where('_.location =~ ".*州"')
    # print(list(persons))
    # ----- 3
    selector = NodeSelector(graph)
    persons = selector.select('Person').order_by('_.age')
    print(list(persons))
    return

def t9():
    # 查询数据
    from py2neo import Node, Relationship, Graph, NodeSelector
    graph = Graph('http://123.59.42.48:8772',
                  username='neo4j',
                  password='huerge4040')
    s = NodeSelector(graph)
    persons = s.select('Hudongitem').where('_.name=~"密度板"')
    print(list(persons))


if __name__ == '__main__':
    # t1()
    # t2()
    # t3()
    # t4()
    # t5()
    # t6()
    # t7()
    t9()
    pass