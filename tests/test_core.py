#!/usr/bin/env python
import unittest
from gts.gtsconfig import GtsConfig

class TestCore(unittest.TestCase):

    def test_config_init(self):        
        conf = GtsConfig()
        conf.loadFromJson('conf_test.json')
        conf.configure()
        return True

    def test_config_txt(self):
        conf = GtsConfig()
        conf.load_subjects('test_subjs.txt')
        subjs = [i.name for i in conf.subjects]
        self.assertEqual(subjs, ['C1','C2','S1','S2'])
    
    def test_config_csv(self):        
        conf = GtsConfig()
        conf.load_subjects('test_subjs.csv')
        subjs = [i.name for i in conf.subjects]
        #print conf.subjects
        self.assertEqual(subjs, ['C1','C2','S1','S2'])




if __name__ == '__main__':
    unittest.main()