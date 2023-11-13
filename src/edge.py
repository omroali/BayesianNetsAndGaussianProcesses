from ConditionalIndependence import ConditionalIndependence as ci

class Edge:
    def __init__(
            self, 
            var_i: str, 
            var_j: str, 
            parents: list[str], 
            conditional_independence_test: ci, 
            remaining_nodes: list[str] = [],
            threshold: float = 0.01,
            # *directional: bool
        ):
        self.var_i = var_i
        self.var_j = var_j 
        self.parents = parents
        self.remaining_nodes = []
        self.conditional_independence_test = conditional_independence_test
        self.independence_threshold = threshold
        self.test_value = None
        # self.directional = False if not directional else directional # implies direction from Vi to Vj if true
        
    ###################################################
    ################ CLASS METHODS ####################
    ###################################################
    
    def is_independent(self, var_i: str, var_j: str, parents: list[str]) -> bool:
        return self.independence_test(self.conditional_independence_test, var_i, var_j, parents, self.independence_threshold)
    
    def is_dependent(self, var_i: str, var_j: str, parents: list[str]) -> bool:
        return not self.independence_test(self.conditional_independence_test, var_i, var_j, parents, self.independence_threshold)
    
    ###################################################
    ################# PROPERTIES ######################
    ###################################################
    
    @property
    def is_multi_parent(self) -> bool:
        '''
        Check if the edge has more than one parent
        '''
        return len(self.parents) > 1
    
    @property
    def has_one_parent(self) -> bool:
        '''
        Check if the edge has more than one parent
        '''
        return len(self.parents) == 1
    
    @property
    def is_parentless(self) -> bool:
        '''
        Check if the edge has more than one parent
        '''
        return len(self.parents) == 0

    @property
    def markov_approved(self) -> bool:
        ''' 
        TODO: RENAME
        this mean it has_markov, has_minimality, has_faithfulness
        '''
        return self.is_conditionally_independent and self.has_minimality and self.has_faithfulness
        
    @property
    def is_conditionally_independent(self) -> bool:
        '''
        Vi - P[] - Vj
        condition V1 independent of V2 given P[]
        '''
        self.test_value = float(self.conditional_independence_test.compute_pvalue(self.var_i, self.var_j, self.parents))
        return self.test_value  > self.independence_threshold
    
    @property
    def has_minimality(self) -> bool:
        '''
        Vi - P[] - Vj
        condition 1 Vi is independent of P[] 
        condition 2 P[] is independent of Vj conditional on Nothing
        '''
        if self.is_multi_parent:
            raise ValueError('ERROR: Only one parent is currently support for current minimality test')
        if self.is_parentless:
            return True
        cond_1 = self.is_independent(self.var_i, self.var_j, [])
        cond_2 = self.is_independent(self.parents[0], self.var_i, [])
        
        return cond_1 and cond_2
    
    @property
    def has_faithfulness(self) -> bool:
        '''
        Vi - P[] - Vj
        condition Vi is independent of Vj
        '''
        return self.is_independent(self.var_i, self.var_j, [])

    @property
    def is_immoral(self) -> bool:
        '''
        condition 1 Vi independent of Vj
        condition 2 Vi dependent of Vj conditional on Par
        '''
        cond_1 = self.is_independent(self.var_i, self.var_j, [])
        cond_2 = self.is_dependent(self.var_i, self.var_j, self.parents)
        
        return cond_1 and cond_2
    
    @property
    def as_list(self) -> list[str|list[str]]:
        return [self.var_i, self.var_j, self.parents]
        
            
    # @property
    # def is_chain(self) -> bool:
    #     '''MAY NOT BE IMPLEMENTABLE UNLESS THE GRAPH IS CONNECTED UP WITH THE IMMORALITIES'''
    #     raise NotImplementedError
    
    # @property
    # def is_fork(self) -> bool:
    #     '''MAY NOT BE IMPLEMENTABLE UNLESS THE GRAPH IS CONNECTED UP WITH THE IMMORALITIES'''
    #     raise NotImplementedError
    
    ###################################################
    ############### STATIC METHODS ####################
    ###################################################
    
    @staticmethod
    def independence_test(ci: ci, var_i: str, var_j: str, parents: list[str], threshold: float) -> bool:
        return float(ci.compute_pvalue(var_i, var_j, parents)) < threshold 
    
    # @staticmethod
    # def has_markov_equivalence(edge1, edge2) -> bool:
    #     '''
    #     2 edges are markov equivalent if:
    #     condition 1: they both have has_markov independence
    #     condition 2: has_minimality
    #     condition 3: has_faithfulness
    #     '''
    #     raise NotImplementedError        