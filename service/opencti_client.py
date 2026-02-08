# from typing import List, Dict
# from pycti import OpenCTIApiClient
# import logging, requests, json
# from collections import Counter
# log = logging.getLogger(__name__)

# class OpenCTIClientWrapper:
#     def __init__(self, url: str, token: str):
#         self.client = OpenCTIApiClient(url, token)
#         self._url = url.rstrip('/'); self._token = token

#     def test_connection(self) -> bool:
#         try:
#             me = self.client.user.me()
#             log.info('Connected to OpenCTI as %s', (me or {}).get('name') or (me or {}).get('user_email') or (me or {}).get('id'))
#             return True
#         except Exception as primary_err:
#             try:
#                 _ = self.client.stix_domain_object.list(types=['Indicator'], first=1)
#                 log.info('Connected to OpenCTI (list test succeeded)'); return True
#             except Exception as e:
#                 log.exception('OpenCTI connection failed: %s ; fallback error: %s', primary_err, e); return False

#     def fetch_all_objects(self) -> List[Dict]:
#         objs = []
#         wanted = ['Indicator','Malware','Intrusion-Set','Threat-Actor-Individual','Report','Campaign','Attack-Pattern','Incident']
#         for t in wanted:
#             try:
#                 items = self.client.stix_domain_object.list(types=[t], first=1000)
#                 for it in items or []:
#                     objs.append({'id': it.get('id'),
#                                  'type': it.get('entity_type') or t,
#                                  'name': it.get('name') or it.get('standard_id') or it.get('id'),
#                                  'confidence': it.get('confidence') or 0,
#                                  'updated_at': it.get('updated_at'),
#                                  'createdBy': (it.get('createdBy') or {}).get('id')})
#             except Exception as ee:
#                 log.warning('Failed fetching type %s: %s', t, ee)
#         log.info('Fetched %d STIX objects', len(objs))
#         log.info('Per-type counts: %s', dict(Counter(o['type'] for o in objs)))
#         return objs

#     def _list_reports(self):
#         try: return self.client.stix_domain_object.list(types=['Report'], first=2000) or []
#         except Exception as e: log.warning('Failed listing reports: %s', e); return []

#     def fetch_report_object_refs(self) -> List[Dict]:
#         rels, reports = [], self._list_reports()
#         if not reports: return rels
#         has_ref = hasattr(self.client,'stix_ref_relationship') and hasattr(self.client.stix_ref_relationship,'list')
#         if has_ref:
#             log.info('Listing stix_ref_relationships (report object refs)')
#             for r in reports:
#                 rid = r.get('id')
#                 try:
#                     items = self.client.stix_ref_relationship.list(fromId=rid, relationship_type='object', first=2000)
#                     for it in items or []:
#                         rels.append({'id': it.get('id'),'type': it.get('relationship_type') or 'object',
#                                      'source_ref': it.get('from',{}).get('id'),'target_ref': it.get('to',{}).get('id'),'confidence': None})
#                 except Exception as e: log.warning('Ref rel list failed for report %s: %s', rid, e)
#             return rels
#         # GraphQL fallback with inline fragments for union
#         log.info('Listing report object refs via GraphQL fallback')
#         graphql_url = f"{self._url}/graphql"
#         headers = {'Authorization': f'Bearer {self._token}','Content-Type':'application/json'}
#         query = '''        query ReportObjectRefs($id: String!, $first: Int) {
#           stixCoreObject(id: $id) {
#             ... on Report {
#               objects(first: $first) {
#                 edges { node {
#                   __typename
#                   ... on StixDomainObject { id }
#                   ... on StixCyberObservable { id }
#                   ... on StixCoreObject { id }
#                   ... on StixObject { id }
#                   ... on BasicObject { id }
#                 }}}
#             }}
#         }'''
#         import requests, json
#         for r in reports:
#             rid = r.get('id')
#             try:
#                 resp = requests.post(graphql_url, headers=headers, data=json.dumps({'query': query,'variables':{'id': rid,'first': 2000}}), timeout=20)
#                 if resp.status_code != 200: log.warning('GraphQL HTTP %s for %s', resp.status_code, rid); continue
#                 payload = resp.json()
#                 if 'errors' in payload: log.warning('GraphQL errors for report %s: %s', rid, payload['errors']); continue
#                 edges = (((payload.get('data') or {}).get('stixCoreObject') or {}).get('objects') or {}).get('edges', [])
#                 for e in edges:
#                     node = (e or {}).get('node') or {}; tgt = node.get('id')
#                     if tgt: rels.append({'id': f'{rid}::object::{tgt}','type':'object','source_ref': rid,'target_ref': tgt,'confidence': None})
#             except Exception as e: log.warning('GraphQL fallback failed for report %s: %s', rid, e)
#         return rels

#     def fetch_all_relationships(self) -> List[Dict]:
#         rels = []
#         try:
#             log.info('Listing stix_core_relationships')
#             items = self.client.stix_core_relationship.list(first=5000)
#             for it in items or []:
#                 rels.append({'id': it.get('id'),'type': it.get('relationship_type'),
#                              'source_ref': it.get('from',{}).get('id'),'target_ref': it.get('to',{}).get('id'),
#                              'confidence': it.get('confidence') or 0})
#         except Exception as e: log.exception('fetch_all_relationships (core) error: %s', e)
#         try:
#             rels.extend(self.fetch_report_object_refs())
#         except Exception as e: log.warning('Fetching report object refs failed: %s', e)
#         log.info('Fetched %d relationships', len(rels)); return rels

#     def update_confidence(self, stix_id: str, confidence: int) -> bool:
#         graphql_url = f"{self._url}/graphql"
#         headers = {'Authorization': f'Bearer {self._token}','Content-Type':'application/json'}
#         edit_input = [{'key':'confidence','value':[str(int(confidence))]}]
#         def call(query, variables):
#             resp = requests.post(graphql_url, headers=headers, data=json.dumps({'query':query,'variables':variables}), timeout=20)
#             if resp.status_code != 200: return False, f"HTTP {resp.status_code}: {resp.text[:500]}"
#             payload = resp.json(); 
#             if 'errors' in payload: return False, payload['errors']
#             return True, payload
#         q_dom = "mutation($id:ID!,$input:[EditInput]!){ stixDomainObjectEdit(id:$id){ fieldPatch(input:$input){ id }}}"
#         ok, msg = call(q_dom, {'id': stix_id, 'input': edit_input})
#         if ok: return True
#         q_rel = "mutation($id:ID!,$input:[EditInput]!){ stixCoreRelationshipEdit(id:$id){ fieldPatch(input:$input){ id }}}"
#         ok2, msg2 = call(q_rel, {'id': stix_id, 'input': edit_input})
#         if ok2: return True
#         log.warning('GraphQL update failed for %s; domain_err=%s ; rel_err=%s', stix_id, msg, msg2); return False

#     def poll_changes(self) -> Dict:
#         return {'objects': self.fetch_all_objects(), 'relationships': self.fetch_all_relationships()}


from typing import List, Dict
from pycti import OpenCTIApiClient
import logging, requests, json
from collections import Counter
log = logging.getLogger(__name__)

class OpenCTIClientWrapper:
    def __init__(self, url: str, token: str):
        self.client = OpenCTIApiClient(url, token)
        self._url = url.rstrip('/'); self._token = token

    def test_connection(self) -> bool:
        try:
            me = self.client.user.me()
            log.info('Connected to OpenCTI as %s', (me or {}).get('name') or (me or {}).get('user_email') or (me or {}).get('id'))
            return True
        except Exception as primary_err:
            try:
                _ = self.client.stix_domain_object.list(types=['Indicator'], first=1)
                log.info('Connected to OpenCTI (list test succeeded)'); return True
            except Exception as e:
                log.exception('OpenCTI connection failed: %s ; fallback error: %s', primary_err, e); return False

    def fetch_all_objects(self) -> List[Dict]:
        objs = []
        wanted = ['Indicator','Malware','Intrusion-Set','Threat-Actor-Individual','Report','Campaign','Attack-Pattern','Incident']
        for t in wanted:
            try:
                items = self.client.stix_domain_object.list(types=[t], first=1000)
                for it in items or []:
                    objs.append({'id': it.get('id'),
                                 'type': it.get('entity_type') or t,
                                 'name': it.get('name') or it.get('standard_id') or it.get('id'),
                                 'confidence': it.get('confidence'),
                                 'updated_at': it.get('updated_at'),
                                 'createdBy': (it.get('createdBy') or {}).get('id')})
            except Exception as ee:
                log.warning('Failed fetching type %s: %s', t, ee)
        log.info('Fetched %d STIX objects', len(objs))
        log.info('Per-type counts: %s', dict(Counter(o['type'] for o in objs)))
        return objs

    def _list_reports(self):
        try: return self.client.stix_domain_object.list(types=['Report'], first=2000) or []
        except Exception as e: log.warning('Failed listing reports: %s', e); return []

    def fetch_report_object_refs(self) -> List[Dict]:
        rels, reports = [], self._list_reports()
        if not reports: return rels
        has_ref = hasattr(self.client,'stix_ref_relationship') and hasattr(self.client.stix_ref_relationship,'list')
        if has_ref:
            log.info('Listing stix_ref_relationships (report object refs)')
            for r in reports:
                rid = r.get('id')
                try:
                    items = self.client.stix_ref_relationship.list(fromId=rid, relationship_type='object', first=2000)
                    for it in items or []:
                        rels.append({'id': it.get('id'),'type': it.get('relationship_type') or 'object',
                                     'source_ref': it.get('from',{}).get('id'),'target_ref': it.get('to',{}).get('id'),'confidence': None})
                except Exception as e: log.warning('Ref rel list failed for report %s: %s', rid, e)
            return rels
        # GraphQL fallback with inline fragments for union
        log.info('Listing report object refs via GraphQL fallback')
        graphql_url = f"{self._url}/graphql"
        headers = {'Authorization': f'Bearer {self._token}','Content-Type':'application/json'}
        query = '''        query ReportObjectRefs($id: String!, $first: Int) {
          stixCoreObject(id: $id) {
            ... on Report {
              objects(first: $first) {
                edges { node {
                  __typename
                  ... on StixDomainObject { id }
                  ... on StixCyberObservable { id }
                  ... on StixCoreObject { id }
                  ... on StixObject { id }
                  ... on BasicObject { id }
                }}}
            }}
        }'''
        for r in reports:
            rid = r.get('id')
            try:
                resp = requests.post(graphql_url, headers=headers, data=json.dumps({'query': query,'variables':{'id': rid,'first': 2000}}), timeout=20)
                if resp.status_code != 200: log.warning('GraphQL HTTP %s for %s', resp.status_code, rid); continue
                payload = resp.json()
                if 'errors' in payload: log.warning('GraphQL errors for report %s: %s', rid, payload['errors']); continue
                edges = (((payload.get('data') or {}).get('stixCoreObject') or {}).get('objects') or {}).get('edges', [])
                for e in edges:
                    node = (e or {}).get('node') or {}; tgt = node.get('id')
                    if tgt: rels.append({'id': f'{rid}::object::{tgt}','type':'object','source_ref': rid,'target_ref': tgt,'confidence': None})
            except Exception as e: log.warning('GraphQL fallback failed for report %s: %s', rid, e)
        return rels

    def fetch_all_relationships(self) -> List[Dict]:
        rels = []
        seen = set()
        try:
            log.info('Listing stix_core_relationships')
            items = self.client.stix_core_relationship.list(first=5000)
            for it in items or []:
                typ = it.get('relationship_type')
                src = (it.get('from') or {}).get('id')
                dst = (it.get('to') or {}).get('id')
                if not (src and dst): continue
                k = (src, dst, typ)
                if k in seen: continue
                seen.add(k)
                rels.append({'id': it.get('id'),'type': typ,
                             'source_ref': src,'target_ref': dst,
                             'confidence': it.get('confidence') or 0})
        except Exception as e: log.exception('fetch_all_relationships (core) error: %s', e)
        try:
            for it in self.fetch_report_object_refs():
                typ = it.get('type') or 'object'
                k = (it['source_ref'], it['target_ref'], typ)
                if k in seen: continue
                seen.add(k)
                rels.append(it)
        except Exception as e: log.warning('Fetching report object refs failed: %s', e)
        log.info('Fetched %d relationships', len(rels)); return rels

    def update_confidence(self, stix_id: str, confidence: int) -> bool:
        graphql_url = f"{self._url}/graphql"
        headers = {'Authorization': f'Bearer {self._token}','Content-Type':'application/json'}
        confidence = max(0, min(100, int(round(confidence))))
        edit_input = [{'key':'confidence','value':[str(confidence)]}]
        def call(query, variables):
            resp = requests.post(graphql_url, headers=headers, data=json.dumps({'query':query,'variables':variables}), timeout=20)
            if resp.status_code != 200: return False, f"HTTP {resp.status_code}: {resp.text[:500]}"
            payload = resp.json(); 
            if 'errors' in payload: return False, payload['errors']
            return True, payload
        q_dom = "mutation($id:ID!,$input:[EditInput]!){ stixDomainObjectEdit(id:$id){ fieldPatch(input:$input){ id }}}"
        ok, msg = call(q_dom, {'id': stix_id, 'input': edit_input})
        if ok: return True
        q_rel = "mutation($id:ID!,$input:[EditInput]!){ stixCoreRelationshipEdit(id:$id){ fieldPatch(input:$input){ id }}}"
        ok2, msg2 = call(q_rel, {'id': stix_id, 'input': edit_input})
        if ok2: return True
        log.warning('GraphQL update failed for %s; domain_err=%s ; rel_err=%s', stix_id, msg, msg2); return False

    def poll_changes(self) -> Dict:
        return {'objects': self.fetch_all_objects(), 'relationships': self.fetch_all_relationships()}
