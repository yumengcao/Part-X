
import copy
from Functional.__tools__ import select_regions, vol
import numpy as np

class partitioning:
    def __init__(self, subregions:dict, 
                 dim_index:int, dim:int, uni_sample: dict, 
                 uni_rob: dict, iteration: int, group_result:dict, grouping: str,
                 group_sample_num: dict, iter_group:int, region_vol: float, part_number:int,
                 ):
        '''
       Partitioning Algorithm
       Parameters:
    #         subregion: region need to be partitioned in dict format for one iteration
    #         part_number (int): how many subnode that needed to be generated by one root
    #         dim_index (int): indicate the dimesion needed to be partitioned
    #         dim(int): dimension
   
    #     Returns:
    #         part_sub (dist): subregions with new serial numbers
        '''
        self.subregions = subregions ##subregions that needed to be partitioned
        self.dim_index = dim_index
        self.dim = dim
        self.uni_sample = uni_sample
        self.uni_rob = uni_rob
        self.iteration = iteration
        self.group_result = group_result
        self.grouping = grouping
        self.group_sample_num = group_sample_num
        self.iter_group = iter_group
        self.region_vol = region_vol
        self.part_number = part_number
        
        
        

    def __condition__(self, sub_index):

        assert len(self.subregions[sub_index][0]) == self.dim
        assert len(self.subregions[sub_index][1]) == self.dim
        


    def partitioning_algorithm(self):

        uni_select_X = {}
        uni_select_Y = {}
        part_sub = {}
        upd_sample_g = {}
        
        for sub_index in self.subregions.keys():
            self.__condition__(sub_index)
            sl_coordinate_upper = self.subregions[sub_index][self.dim_index][1]
            sl_coordinate_lower = self.subregions[sub_index][self.dim_index][0]
           
            # if self.iteration > self.iter_group and self.grouping != '0' :
            #     #if vol(self.subregions[sub_index], self.dim) < 0.125* self.region_vol:
            #     if sub_index in self.group_result['group1'] or sub_index in self.group_result['group6']:#\
            #         #or sub_index in self.group_result['group5']:
            #         non_part_series = str(eval(sub_index)*self.part_number)
            #         part_sub[non_part_series] = self.subregions[sub_index]
            #         uni_select_X[non_part_series] = self.uni_sample[sub_index]
            #         uni_select_Y[non_part_series] = self.uni_rob[sub_index]
            #         upd_sample_g[non_part_series] = self.group_sample_num[sub_index]
            #         continue

            for j in range(self.part_number): 
                l_coordinate_lower = float((sl_coordinate_upper - sl_coordinate_lower))* j / self.part_number+ \
                                                        sl_coordinate_lower
                l_coordinate_upper = float((sl_coordinate_upper- sl_coordinate_lower) * (j + 1)) / \
                                                        self.part_number + sl_coordinate_lower
                
                sub_series = str(eval(sub_index)*self.part_number + j - (self.part_number - 1))
                part_sub[sub_series]= copy.deepcopy(self.subregions[sub_index])
                part_sub[sub_series][self.dim_index] = [l_coordinate_lower, l_coordinate_upper]
                
                
                if self.iteration != 0:
                    #print('series:',sub_series)
                    uni_select_X[sub_series], uni_select_Y[sub_series] = select_regions(self.uni_sample[sub_index],
                                                        part_sub[sub_series], self.uni_rob[sub_index], 
                                                        self.dim)
                    if self.grouping == '1' and self.iteration > self.iter_group:
                        upd_sample_g[str(sub_series)] = self.group_sample_num[sub_index]
                    
                    # if self.grouping == '1' and self.iteration > self.iter_group:
                    #     #if vol(self.subregions[sub_index], self.dim) <= 0.125* self.region_vol:
                    #     upd_sample_g[str(sub_series)] = self.group_sample_num[sub_index]
                    
      
            
            
        return part_sub, uni_select_X, uni_select_Y, upd_sample_g
            
