'''
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

Author: Rong Gong, MTG-UPF, rong.gong@upf.edu
17 January 2016
'''

import raw_feature.textgrid.textgridParser as textgridParser

groundtruth_textgrid_file   = 'F:/项目/花城音乐项目/音符起始点检测/part1/textgrid/laosheng/lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf.TextGrid'

# parse the phrase boundary, and its content
lineList            		= textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='line')

# parse the dian Tier
dianList 	                = textgridParser.textGrid2WordList(groundtruth_textgrid_file, whichTier='dian')

onset_times = [tup[0] for tup in dianList if tup[2] != '']
print(lineList)
print(dianList)
print(onset_times)