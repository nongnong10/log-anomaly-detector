"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
from collections import defaultdict
import json

from database.get_events import get_all_event_templates
from database.upsert_log_line import upsert_log_lines  # New import


class Logcluster:
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4,
                 maxChild=100, rex=[], keep_para=True):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:

            # Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']

                else:
                    if '<*>' not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD['<*>']

            # If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    # seq1 is template
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue #comment@haixuanguo: <*> == <*> are similar pairs
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar

    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, logClustL):
        total_lines = self.df_log.shape[0]
        log_templates = [0] * total_lines
        log_templateids = [0] * total_lines
        df_events = []
        for logClust in logClustL:
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]

            # keep only valid LineIds for this run
            valid_ids = []
            for logID in logClust.logIDL:
                idx = logID - 1
                if 0 <= idx < total_lines:
                    log_templates[idx] = template_str
                    log_templateids[idx] = template_id
                    valid_ids.append(logID)

            # update cluster ids to avoid growth with stale data
            logClust.logIDL = valid_ids
            occurrence = len(valid_ids)
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False,
                        columns=["EventId", "EventTemplate", "Occurrences"])

    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def load_previous_clusters(self, templates_csv_path, _structured_csv_path_ignored=None):
        """
        Warm start using only templates CSV.
        templates_csv_path: file with columns EventId, EventTemplate, Occurrences
        LineId info is ignored; clusters start with empty logIDL lists.
        """
        if not os.path.exists(templates_csv_path):
            return []
        df_templates = pd.read_csv(templates_csv_path)
        clusters = []
        seen = set()
        for _, row in df_templates.iterrows():
            template_str = str(row["EventTemplate"])
            if template_str in seen:
                continue
            seen.add(template_str)
            tokens = template_str.split()
            # Start with empty logIDL so new file's lines populate occurrences
            clusters.append(Logcluster(logTemplate=tokens, logIDL=[]))
        return clusters

    def parse(self, logName, previous_templates_csv=None, previous_structured_csv=None):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()

        # Warm start: load old clusters (ignore previous_structured_csv)
        if previous_templates_csv:
            old_clusters = self.load_previous_clusters(previous_templates_csv)
            for c in old_clusters:
                logCluL.append(c)
                self.addSeqToPrefixTree(rootNode, c)

        count = 0
        for idx, line in self.df_log.iterrows():

            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            # logmessageL = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
            matchCluster = self.treeSearch(rootNode, logmessageL)

            # Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            # Add the new log message to the existing cluster
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)), end='\r')

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(logCluL)

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def parse_and_store_log_lines(self, logName, warm_start=None, db_conn=None):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []
        blk_pattern = re.compile(r'(blk_-?\d+)')
        line_to_block = {}
        anomaly_line = {}
        anomaly_template = []
        block_ids = []

        self.load_data()

        # 1. Warm start to load old clusters
        event_templates = []
        if warm_start is not None and warm_start:
            event_templates = get_all_event_templates(db_conn)
            for event in event_templates:
                template_tokens = self.preprocess(event.event_template).strip().split()
                match_cluster = self.treeSearch(rootNode, template_tokens)
                # print(f"Cluster {match_cluster}: Message {template_tokens}")
                if match_cluster is None:
                    new_cluster = Logcluster(logTemplate=template_tokens, logIDL=[])
                    logCluL.append(new_cluster)
                    self.addSeqToPrefixTree(rootNode, new_cluster)

        count = 0
        # 2. Check if any log line is anomaly by clustering content template
        for idx, line in self.df_log.iterrows():
            # print(f"=== Line {line['LineId']}: {line['Content']}")
            # print(f"Preprocess log: {self.preprocess(line['Content'])}")
            # print(f"Tokens: {self.preprocess(line['Content']).strip().split()} ")
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            matchCluster = self.treeSearch(rootNode, logmessageL)
            # print(f"Cluster {matchCluster}: Message {logmessageL}")
            # Match no existing log cluster
            if matchCluster is None:
                print(f"[ANOMALY LOG LINE]: Message {logmessageL}")
                anomaly_line[logID] = True
                new_cluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(new_cluster)
                anomaly_template.append(new_cluster)
                self.addSeqToPrefixTree(rootNode, new_cluster)
            else:
                if matchCluster in anomaly_template:
                    print(f"[ANOMALY LOG LINE]: Message {logmessageL}")
                    anomaly_line[logID] = True
                    matchCluster.logIDL.append(logID)
                else:
                    newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                    matchCluster.logIDL.append(logID)
                    if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                        matchCluster.logTemplate = newTemplate

            # Capture first block_id only
            m = blk_pattern.search(line['Content'])
            if m:
                line_to_block[logID] = m.group(1)
                if m.group(1) not in block_ids:
                    block_ids.append(m.group(1))
            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)), end='\r')

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(logCluL)  # Ensures EventId column populated

        # Build per-block log line payloads (DB logic delegated)
        if db_conn is not None and line_to_block:
            block_payloads = defaultdict(list)
            # Iterate rows again to assemble full info after EventId assignment
            for _, row in self.df_log.iterrows():
                bid = line_to_block.get(row['LineId'])
                if not bid:
                    continue
                print(f"Log line {row['LineId']} - is_anomaly: {anomaly_line.get(row['LineId'])}")
                block_payloads[bid].append({
                    'pid': row.get('Pid') or row.get('PID'),
                    'level': row.get('Level', 'INFO'),
                    'component': row.get('Component', 'UNKNOWN'),
                    'content': row.get('Content', ''),
                    'event_id': row.get('EventId', ''),
                    'is_anomaly': anomaly_line.get(row['LineId'], False)
                })
            # Upsert each block via helper (all SQL lives in upsert_log_line.py)
            for bid, lines in block_payloads.items():
                upsert_log_lines(db_conn, bid, lines)
            print(f"\nUpserted {sum(len(v) for v in block_payloads.values())} log lines across {len(block_payloads)} blocks.")

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        return block_ids

    def parse_to_event_sequences(self, log_file, templates_csv, mapping_json, sequence_output_path):
        """
        Warm start using templates CSV, map EventId to integers, output block-level event sequences.
        Each line in the output file is a space-separated sequence of mapped event integers for one BlockId.
        """
        # 1. Prepare parser state
        self.logName = log_file
        root = Node()
        clusters = []

        # 2. Load warm templates
        warm_clusters = self.load_previous_clusters(templates_csv)
        for c in warm_clusters:
            clusters.append(c)
            self.addSeqToPrefixTree(root, c)

        # 3. Load raw log data
        self.load_data()  # populates self.df_log with columns including Content

        # 4. Load mapping JSON (EventId -> int)
        with open(mapping_json, "r") as f:
            id_map = json.load(f)

        # 5. Parse lines, assign to clusters
        for _, row in self.df_log.iterrows():
            line_id = row["LineId"]
            tokens = self.preprocess(row["Content"]).strip().split()
            match = self.treeSearch(root, tokens)
            if match is None:
                new_c = Logcluster(logTemplate=tokens, logIDL=[line_id])
                clusters.append(new_c)
                self.addSeqToPrefixTree(root, new_c)
            else:
                new_template = self.getTemplate(tokens, match.logTemplate)
                match.logIDL.append(line_id)
                if ' '.join(new_template) != ' '.join(match.logTemplate):
                    match.logTemplate = new_template

        # 6. Assign EventTemplate / EventId for each line
        # Build lookup: template string -> EventId
        template_cache = {}
        line_event_ids = []
        for _, row in self.df_log.iterrows():
            # Reconstruct template by finding cluster containing this LineId
            eid = ''
            templ = ''
            for c in clusters:
                if row["LineId"] in c.logIDL:
                    templ = ' '.join(c.logTemplate)
                    eid = template_cache.get(templ)
                    if eid is None:
                        eid = hashlib.md5(templ.encode('utf-8')).hexdigest()[0:8]
                        template_cache[templ] = eid
                    break
            line_event_ids.append(eid)

        self.df_log["EventId"] = line_event_ids

        # 7. Build block-level sequences
        block_sequences = defaultdict(list)
        blk_pattern = re.compile(r'(blk_-?\d+)')
        for _, r in self.df_log.iterrows():
            eid = r["EventId"]
            mapped = id_map.get(eid, -1)
            if mapped == -1:
                continue  # skip unmapped events
            blk_ids = set(blk_pattern.findall(r["Content"]))
            for b in blk_ids:
                block_sequences[b].append(mapped)

        # 8. Write sequences file (each line: space-separated integers)
        with open(sequence_output_path, 'w') as fout:
            for _, seq in block_sequences.items():
                if seq:
                    fout.write(' '.join(str(x) for x in seq) + '\n')

        return sequence_output_path

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        cnt = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    # print("\n", line)
                    # print(e)
                    pass
        print("Total size after encoding is", linecount, cnt)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", str(row["EventTemplate"]))
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r' +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list
