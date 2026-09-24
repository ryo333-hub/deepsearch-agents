"""Opt-in local MySQL ecommerce data + SELECT-only account acceptance.

RUN_ECOMMERCE_INTEGRATION=1 enables this suite independently of the legacy
RUN_LOCAL_SERVICE_INTEGRATION tests. No default .env or pharma data is changed.
"""

import os
import unittest

from scripts.verify_ecommerce_demo import verify_mysql


@unittest.skipUnless(os.getenv('RUN_ECOMMERCE_INTEGRATION')=='1',
                     'integration: requires explicitly provisioned ecommerce database')
class EcommerceDataIntegrationTests(unittest.TestCase):
    integration=True

    def test_real_data_features_and_database_permission_boundary(self):
        result=verify_mysql()
        self.assertTrue(result['persisted_rows_equal_generated_data'])
        self.assertTrue(result['unchanged_after_permission_probes'])
        self.assertEqual(set(result['permission_probes']),{'INSERT','UPDATE','DELETE'})
        self.assertTrue(all(r['mysql_errno']==1142 for r in result['permission_probes'].values()))
