import os
import sys
import json
import xlrd
from docx import Document


def parse_xlsx(xlsx_path):
    workbook = xlrd.open_workbook(xlsx_path, 'rb')  # , formatting_info=True)
    qid_list_dict = {}
    cid_cnt_dict = {}  # count lines for each paragraph
    for sheet_idx in range(0, 1):
        sheet = workbook.sheet_by_index(sheet_idx)
        for row_idx in range(1, sheet.nrows):
            items = sheet.row_values(row_idx)
            if len(items) == 9 and items[8] == '@@@@':  # if answer is in title, ignore
                continue
            items = items[:8]
            if items == [''] * 8:  # if empty line, ignore
                continue
            print(items)
            # xf_list = workbook.xf_list
            # s = xf_list[0]
            # s = s.background
            # xf = workbook.xf_list[xxx]
            # bgx = xf.background.pattern_colour_index
            # print(bgx)
            for col_idx in range(len(items)):
                if isinstance(items[col_idx], str):
                    items[col_idx] = items[col_idx].strip()
                if col_idx == 3:  # qid
                    items[col_idx] = items[2] + '%04d' % int(items[col_idx])  # get question id
                if col_idx == 6:
                    splits = items[col_idx].split('\n')
                    answer_list = []
                    for split in splits:
                        split = split.strip()
                        if split == '':
                            continue
                        answer_list.append(split)
                    items[col_idx] = '\n'.join(answer_list)
            qid_list_dict[items[3]] = items  # [customer, docName, cid, qid, ques_type, ques, ans, yes/no]
            cid_cnt_dict[items[2]] = cid_cnt_dict.get(items[2], 0) + 1
    workbook.release_resources()
    return qid_list_dict, cid_cnt_dict


def parse_docx(docx_path):
    document = Document(docx_path)
    ps = document.paragraphs
    ps_details = [[x.text.strip(), x.style.name.strip(), list(x.runs)] for x in ps]
    cid_ans_idx_dict = {}  # {cid: [[answer, start_idx]]}
    cid_list_dict = {}  # {cid: [title, text]}
    cid = ''
    ques_cnt = 1
    for ps_detail in ps_details:
        # print('docx text: {}'.format(ps_detail[0]))
        # if empty line, invalid title or catalog, ignore
        if ps_detail[0] == '' or ps_detail[0].startswith('@@@@') or \
                (not ps_detail[0].startswith('####') and cid == ''):
            continue
        if ps_detail[0].startswith('####'):  # format: ####[userId][docId][paragraphId]#### title
            ques_cnt = 1
            cid = ps_detail[0][4:].strip()[:12]
            title = ps_detail[0][4:].strip()[12:].strip().lstrip('####').strip()
            # cid_list_dict[cid] = [title, title + '\n']
            print('cid: {}\ttitle: {}'.format(cid, title))
            cid_list_dict[cid] = [title, '']
            continue
            # if consider the situation that answer is underlined in title, use following code
            # ps_detail[0] = title
            # new_ps_detail_2 = []
            # for run_idx in range(len(ps_detail[2])-1, -1, -1):
            #     run_text = ps_detail[2][run_idx].text.strip()
            #     ridx = title.rfind(run_text)
            #     if ridx == -1:
            #         break
            #     new_ps_detail_2.insert(0, ps_detail[2][run_idx])
            #     title = title[:ridx]
            # ps_detail[2] = new_ps_detail_2

        # find underline text position
        start_idx = 0
        last_paragraphs_tail_idx = len(cid_list_dict[cid][1])
        cur_paragraph = ps_detail[0]
        answer_flag = False
        for run in ps_detail[2]:
            run_text = run.text.strip()
            start_idx += cur_paragraph[start_idx:].index(run_text)
            if run.underline:
                print(run.text.strip())
                if answer_flag is False:
                    qid = cid + '%04d' % ques_cnt
                    print('qid: {}\tans_start_idx: {}'.format(qid, last_paragraphs_tail_idx + start_idx))
                    if cid not in cid_ans_idx_dict.keys():
                        cid_ans_idx_dict[cid] = []
                    cid_ans_idx_dict[cid].append([run_text, last_paragraphs_tail_idx + start_idx])
                    ques_cnt += 1
                answer_flag = True
            else:
                answer_flag = False
        cid_list_dict[cid][1] += ps_detail[0] + '\n'
    return cid_list_dict, cid_ans_idx_dict


def get_result(cid_list_dict, qid_list_dict, qid_idx_dict):  # docx, xlsx, docx
    docid_result_dict = {}  # {[userId][docId]: result_list}
    qids = sorted(qid_list_dict.keys())
    last_idx = 0
    last_cid = ''
    for qid in qids:
        xlsx_info = qid_list_dict[qid]
        cid = qid[:12]
        if cid != last_cid:
            last_idx = 0
        title = cid_list_dict[cid][0]
        paragraph = cid_list_dict[cid][1]
        question = xlsx_info[5]
        answer = xlsx_info[6]
        json_ques = {
            'question_id': qid, 'question_type': xlsx_info[4], 'question': question,
            'segmented_question': '', 'questions': [], 'segmented_questions': [],
            'answers': [answer], 'segmented_answers': [], 'answer_docs': [0],
            'answer_spans': [[0, 0]], 'match_scores': [1.00], 'fake_answers': [answer],
            'yesno_answers': [xlsx_info[7]],
            'customer': xlsx_info[0], 'doc_name': xlsx_info[1], 'cid': cid
        }
        json_doc = {
            'paragraphs': [paragraph], 'segmented_paragraphs': [], 'title': title,
            'segmented_title': [''], 'bs_rank_pos': 0, 'is_select': True,
            'most_related_para': 0, 'span': (0, 0)
        }

        ans_idx_list = qid_idx_dict[cid]
        if len(ans_idx_list) == 0:
            aidx = last_idx + paragraph[last_idx:].find(answer)
            if aidx == last_idx + -1:
                print('Error! Cannot find qid: {}\nquestion: {}\nanswer: {}\ntext: {}'.format(
                    qid, question, answer, paragraph))
                sys.exit(-1)
        else:
            if answer.startswith(ans_idx_list[0][0]):
                aidx = ans_idx_list[0][1]
                qid_idx_dict[cid] = qid_idx_dict[cid][1:]
            else:
                aidx = last_idx + paragraph[last_idx:].find(answer)
                if aidx == last_idx + -1:
                    print('Error! Cannot find qid: {}\nquestion: {}\nanswer: {}\ntext: {}'.format(
                        qid, question, answer, paragraph))
                    sys.exit(-1)
        last_idx = aidx + len(answer)

        json_ques['answer_span'] = [[aidx, last_idx]]
        json_ques['documents'] = [json_doc]
        docid = cid[:8]
        if docid not in docid_result_dict.keys():
            docid_result_dict[docid] = []
        docid_result_dict[docid].append(json_ques)
    return docid_result_dict


def write_json(data_dir, docid_result_dict):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    docids = sorted(docid_result_dict.keys())
    for docid in docids:
        json_path = os.path.join(data_dir, docid + '.json')
        with open(json_path, 'w', encoding='utf-8') as fw:
            fw.write(json.dumps(docid_result_dict[docid], ensure_ascii=False))


def write_stat(cid_cnt_dict, file_path):
    cids = sorted(cid_cnt_dict.keys())
    docid_cnt_map = {}
    for cid in cids:
        docid_cnt_map[cid[:8]] = docid_cnt_map.get(cid[:8], 0) + cid_cnt_dict[cid]
    docids = sorted(docid_cnt_map.keys())
    total_cnt = 0
    with open(file_path, 'w', encoding='utf-8') as fw:
        for docid in docids:
            fw.write('docId: {}\tcount: {}\n'.format(docid, docid_cnt_map[docid]))
            total_cnt += docid_cnt_map[docid]
        fw.write('Total count: {}\n'.format(total_cnt))


def batch_process(dir_name):
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)
    userid_list = sorted(os.listdir(main_dir))
    cid_cnt_map = {}
    for userid in userid_list:
        if userid == 'json' or userid.endswith('.zip') or userid.endswith('.txt'):
            continue
        userid_dir = os.path.join(main_dir, userid)
        filenames = sorted(os.listdir(userid_dir))
        qid_list_map = {}
        cid_list_map = {}
        qid_idx_map = {}
        for filename in filenames:
            if filename.startswith('~'):
                continue
            file_path = os.path.join(userid_dir, filename)
            print('file_path: {}'.format(file_path))
            if file_path.endswith('xlsx'):
                sub_qid_list_map, sub_cid_cnt_map = parse_xlsx(file_path)
                qid_list_map.update(sub_qid_list_map)
                cid_cnt_map.update(sub_cid_cnt_map)
            elif file_path.endswith('docx'):
                sub_cid_list_map, sub_qid_idx_map = parse_docx(file_path)
                cid_list_map.update(sub_cid_list_map)
                qid_idx_map.update(sub_qid_idx_map)
            else:
                print('Error! Invalid file type: {}'.format(file_path))
                sys.exit(-1)
        print('start combine xlsx and docx. userid: {}'.format(userid))
        docid_result_map = get_result(cid_list_map, qid_list_map, qid_idx_map)
        write_json(os.path.join(os.path.join(main_dir, 'json')), docid_result_map)
        print('process userid: {} ok!'.format(userid))
    write_stat(cid_cnt_map, os.path.join(main_dir, 'stat.txt'))
    print('Done.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python docx_test.py dir_name')
        sys.exit(-1)
    batch_process(sys.argv[1])
    # main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '20180711')
    # parse_docx(os.path.join(os.path.join(main_dir, '0002'), '0042a3723632-5f75-4408-bdba-6b5443d65b35_terms.docx'))

