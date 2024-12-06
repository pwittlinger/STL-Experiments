# -*- coding: utf-8 -*-
"""Declare2STL Parser

Automatically generated by Colab.


#Quick Start


Below is the Parser class. It takes "cDeclare" templates (see paper) and can return a String our STLCG output.

"""

import re
import sys
sys.path.insert(0, 'src')
import stlcg as stlcg
from stlcg import Expression
import torch
import numpy as np

class Parser:


    def __init__(self):
      # Dictionary to store template information
      self.catalogue = {
        "Existence": {"num_intervals": 1, "num_parameters": 1},
        "PeriodExistence": {"num_intervals": 2, "num_parameters": 1},
        "Response": {"num_intervals": 2, "num_parameters": 2},
        "RespondedExistence": {"num_intervals": 2, "num_parameters": 2},
        "CoExistence": {"num_intervals": 1, "num_parameters": 2},
        "Choice": {"num_intervals": 1, "num_parameters": 2},
        "Absence": {"num_intervals": 1, "num_parameters": 1},
        "NotPeriodExistence": {"num_intervals": 2, "num_parameters": 1},
        "NotResponse": {"num_intervals": 2, "num_parameters": 2},
        "NotCoExistence": {"num_intervals": 1, "num_parameters": 2},
      }


    #takes constraint and extracts interval(s)
    def get_constraint_intervals_as_list(self, template_type, constraint_str):

      if template_type not in self.catalogue:
        raise ValueError(f"Unknown template type: {template_type}")

      # Determine the expected number of intervals
      num_intervals = self.catalogue[template_type]["num_intervals"]

      # Match intervals in square brackets, e.g., [0,1]
      intervals = re.findall(r'\[([^\[\],]+),([^\[\],]+)\]', constraint_str)

      # Convert to list of integer pairs
      parsed_intervals = [[start, end] for start, end in intervals]

      # Validate the number of intervals
      if len(parsed_intervals) != num_intervals:
        raise ValueError(f"Expected {num_intervals} intervals for template type '{template_type}', but found {len(parsed_intervals)}.")

      #returns a list of intervals, where each interval is a list [start, end].
      return parsed_intervals


    #takes constraint and extracts parameter(s)
    def get_constraint_parameters_as_list(self, template_type, constraint_str):
      if template_type not in self.catalogue:
        raise ValueError(f"Unknown template type: {template_type}")

      # Determine the expected number of parameters
      num_parameters = self.catalogue[template_type]["num_parameters"]

      # Match parameters inside parentheses, e.g., (a[t]>100) or (a[t]>100, b[t]<50)
      parameters = re.search(r'\((.*)\)', constraint_str).group(1)

      # Split the parameters into individual components if there are two
      param_list = [param.strip() for param in parameters.split(',')] if num_parameters == 2 else [parameters.strip()]

      # Check if the number of parameters matches the expected count
      if len(param_list) != num_parameters:
        raise ValueError(f"Expected {num_parameters} parameters for template type '{template_type}', but found {len(parameters)}.")


      return param_list


    #takes a list of parameters and decomposes the primitives ("a", "<", "100"). needed to construct stlcg objects
    def get_parameter_primitives(self, param_list):
      processed_params = []

      for param in param_list:
        # If the parameter contains "dis", extract only the variable letter
        if 'dis' in param:
            variable = re.search(r'dis\((\w)\[t\]\)', param)  # Extract the variable letter
            if variable:
                processed_params.append([variable.group(1)])  # Only return the variable letter
        else:
            # Otherwise, parse the variable, comparator, and constant
            match = re.match(r'([a-zA-Z]+)\[t\]\s*([<>=!]+)\s*(\d+)', param)  # Match variable[t] <const
            if match:
                variable = match.group(1)
                comparator = match.group(2)
                constant = match.group(3)
                processed_params.append([variable, comparator, constant])

      return processed_params

    #small helper for constructing stlcg terms
    #todo: it seems stlccg does not support < and >
    def stlcg_term_helper(self, variable, comparator, constant):
      if comparator == "<":
        return stlcg.LessThan(lhs=variable, val=float(constant))
      elif comparator == "<=":
        return stlcg.LessThan(lhs=variable, val=float(constant))
      elif comparator == "=":
        return stlcg.Equal(lhs=variable, val=float(constant))
      elif comparator == ">":
        return stlcg.GreaterThan(lhs=variable, val=float(constant))
      elif comparator == ">=":
        return stlcg.GreaterThan(lhs=variable, val=float(constant))

    def stlcg_interval_helper(self, interval_a, interval_b):
      if interval_b == "m": #infinity
        return None #defaults to [0,np.inf]
      else:
        return [int(interval_a),int(interval_b)]




    #===================================
    #
    #transforms a declare string to STLCG object (returns stlcg object)
    #
    #===================================
    def transform_declare2STLCG(self, constraint_str):
      #TODO: check for cases when last parameter is m

      template_type = constraint_str.split("[", 1)[0]
      interval_list = self.get_constraint_intervals_as_list(template_type, constraint_str)
      param_list = self.get_constraint_parameters_as_list(template_type, constraint_str)
      param_primitives_list = self.get_parameter_primitives(param_list)

      stlcg_object = None;

      #Existence
      if template_type == "Existence":
        isDiscrete = len(param_primitives_list[0])==1
        variable = param_primitives_list[0][0]
        comparator = '=' if (isDiscrete) else param_primitives_list[0][1]
        constant = 1 if (isDiscrete) else param_primitives_list[0][2]

        k = torch.tensor(float(constant), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable, comparator, constant)

        #f2 = stlcg.Eventually(subformula=f1, interval=[ int(interval_list[0][0]), int(interval_list[0][1])] )
        f2 = stlcg.Eventually(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]) )


        stlcg_object = f2

      # PeriodExistence Template
      elif template_type == "PeriodExistence":
        isDiscrete = len(param_primitives_list[0]) == 1
        variable = param_primitives_list[0][0]
        comparator = '=' if isDiscrete else param_primitives_list[0][1]
        constant = 1 if isDiscrete else param_primitives_list[0][2]

        k = torch.tensor(float(constant), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable, comparator, constant)

        f2 = stlcg.Always(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))
        f3 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[1][0], interval_list[1][1]))

        stlcg_object = f3

      # Response Template
      elif template_type == "Response":
        isDiscrete_a = len(param_primitives_list[0]) == 1
        variable_a = param_primitives_list[0][0]
        comparator_a = '=' if isDiscrete_a else param_primitives_list[0][1]
        constant_a = 1 if isDiscrete_a else param_primitives_list[0][2]

        k_a = torch.tensor(float(constant_a), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable_a, comparator_a, constant_a)

        isDiscrete_r = len(param_primitives_list[1]) == 1
        variable_r = param_primitives_list[1][0]
        comparator_r = '=' if isDiscrete_r else param_primitives_list[1][1]
        constant_r = 1 if isDiscrete_r else param_primitives_list[1][2]

        k_r = torch.tensor(float(constant_r), dtype=torch.float, requires_grad=False)
        f2 = self.stlcg_term_helper(variable_r, comparator_r, constant_r)

        f3 = stlcg.Always(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))
        f4 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[1][0], interval_list[1][1]))

        stlcg_object = stlcg.Implies(f3, f4)

      # RespondedExistence Template
      elif template_type == "RespondedExistence":
        isDiscrete_a = len(param_primitives_list[0]) == 1
        variable_a = param_primitives_list[0][0]
        comparator_a = '=' if isDiscrete_a else param_primitives_list[0][1]
        constant_a = 1 if isDiscrete_a else param_primitives_list[0][2]

        k_a = torch.tensor(float(constant_a), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable_a, comparator_a, constant_a)

        isDiscrete_r = len(param_primitives_list[1]) == 1
        variable_r = param_primitives_list[1][0]
        comparator_r = '=' if isDiscrete_r else param_primitives_list[1][1]
        constant_r = 1 if isDiscrete_r else param_primitives_list[1][2]

        k_r = torch.tensor(float(constant_r), dtype=torch.float, requires_grad=False)
        f2 = self.stlcg_term_helper(variable_r, comparator_r, constant_r)

        f3 = stlcg.Eventually(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))
        f4 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[1][0], interval_list[1][1]))

        stlcg_object = stlcg.Implies(f3, f4)

      # CoExistence Template
      elif template_type == "CoExistence":
        isDiscrete_1 = len(param_primitives_list[0]) == 1
        variable_1 = param_primitives_list[0][0]
        comparator_1 = '=' if isDiscrete_1 else param_primitives_list[0][1]
        constant_1 = 1 if isDiscrete_1 else param_primitives_list[0][2]

        k_1 = torch.tensor(float(constant_1), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable_1, comparator_1, constant_1)

        isDiscrete_2 = len(param_primitives_list[1]) == 1
        variable_2 = param_primitives_list[1][0]
        comparator_2 = '=' if isDiscrete_2 else param_primitives_list[1][1]
        constant_2 = 1 if isDiscrete_2 else param_primitives_list[1][2]

        k_2 = torch.tensor(float(constant_2), dtype=torch.float, requires_grad=False)
        f2 = self.stlcg_term_helper(variable_2, comparator_2, constant_2)

        f3 = stlcg.Eventually(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))
        f4 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))

        f5 = stlcg.Implies(f3, f4)
        f6 = stlcg.Implies(f4,f3)


        stlcg_object = stlcg.And(f5, f6)

      # Choice Template
      elif template_type == "Choice":
        isDiscrete_1 = len(param_primitives_list[0]) == 1
        variable_1 = param_primitives_list[0][0]
        comparator_1 = '=' if isDiscrete_1 else param_primitives_list[0][1]
        constant_1 = 1 if isDiscrete_1 else param_primitives_list[0][2]

        k_1 = torch.tensor(float(constant_1), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable_1, comparator_1, constant_1)

        isDiscrete_2 = len(param_primitives_list[1]) == 1
        variable_2 = param_primitives_list[1][0]
        comparator_2 = '=' if isDiscrete_2 else param_primitives_list[1][1]
        constant_2 = 1 if isDiscrete_2 else param_primitives_list[1][2]

        k_2 = torch.tensor(float(constant_2), dtype=torch.float, requires_grad=False)
        f2 = self.stlcg_term_helper(variable_2, comparator_2, constant_2)

        f3 = stlcg.Eventually(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))
        f4 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))

        stlcg_object = stlcg.Or(f3, f4)

      # Absence Template
      elif template_type == "Absence":
        isDiscrete = len(param_primitives_list[0]) == 1
        variable = param_primitives_list[0][0]
        comparator = '=' if isDiscrete else param_primitives_list[0][1]
        constant = 1 if isDiscrete else param_primitives_list[0][2]

        k = torch.tensor(float(constant), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable, comparator, constant)
        f2 = stlcg.Eventually(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))

        stlcg_object = stlcg.Negation(f2)

      # NotPeriodExistence Template
      elif template_type == "NotPeriodExistence":
        isDiscrete = len(param_primitives_list[0]) == 1
        variable = param_primitives_list[0][0]
        comparator = '=' if isDiscrete else param_primitives_list[0][1]
        constant = 1 if isDiscrete else param_primitives_list[0][2]

        k = torch.tensor(float(constant), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable, comparator, constant)


        f2 = stlcg.Always(subformula=f1, interval=self.stlcg_interval_helper(interval_list[1][0], interval_list[1][1]))
        f4 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))

        stlcg_object = f4

      # NotResponse Template
      elif template_type == "NotResponse":
        isDiscrete_a = len(param_primitives_list[0]) == 1
        variable_a = param_primitives_list[0][0]
        comparator_a = '=' if isDiscrete_a else param_primitives_list[0][1]
        constant_a = 1 if isDiscrete_a else param_primitives_list[0][2]

        k_a = torch.tensor(float(constant_a), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable_a, comparator_a, constant_a)

        isDiscrete_r = len(param_primitives_list[1]) == 1
        variable_r = param_primitives_list[1][0]
        comparator_r = '=' if isDiscrete_r else param_primitives_list[1][1]
        constant_r = 1 if isDiscrete_r else param_primitives_list[1][2]

        k_r = torch.tensor(float(constant_r), dtype=torch.float, requires_grad=False)
        f2 = self.stlcg_term_helper(variable_r, comparator_r, constant_r)

        f3 = stlcg.Always(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))
        f4 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[1][0], interval_list[1][1]))

        stlcg_object = stlcg.Implies(f3, stlcg.Negation(f4))

      # NotCoExistence Template
      elif template_type == "NotCoExistence":
        isDiscrete_1 = len(param_primitives_list[0]) == 1
        variable_1 = param_primitives_list[0][0]
        comparator_1 = '=' if isDiscrete_1 else param_primitives_list[0][1]
        constant_1 = 1 if isDiscrete_1 else param_primitives_list[0][2]

        k_1 = torch.tensor(float(constant_1), dtype=torch.float, requires_grad=False)
        f1 = self.stlcg_term_helper(variable_1, comparator_1, constant_1)

        isDiscrete_2 = len(param_primitives_list[1]) == 1
        variable_2 = param_primitives_list[1][0]
        comparator_2 = '=' if isDiscrete_2 else param_primitives_list[1][1]
        constant_2 = 1 if isDiscrete_2 else param_primitives_list[1][2]

        k_2 = torch.tensor(float(constant_2), dtype=torch.float, requires_grad=False)
        f2 = self.stlcg_term_helper(variable_2, comparator_2, constant_2)

        f3 = stlcg.Eventually(subformula=f1, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))
        f4 = stlcg.Eventually(subformula=f2, interval=self.stlcg_interval_helper(interval_list[0][0], interval_list[0][1]))

        stlcg_object = stlcg.Negation(stlcg.And(f3, f4))

      else:
        raise ValueError(f"Unknown template type: {template_type}")


      return stlcg_object




    #===================================
    #
    #transforms a declare string to STL(returns text)
    #
    #===================================
    def transform_declare2text(self, constraint_str):

      template_type = constraint_str.split("[", 1)[0]
      interval_list = self.get_constraint_intervals_as_list(template_type, constraint_str)
      param_list = self.get_constraint_parameters_as_list(template_type, constraint_str)


      stl_semantics = ""

      if template_type == "Existence":
        stl_semantics = f"F[{interval_list[0][0]},{interval_list[0][1]}]({param_list[0]})"

      elif template_type == "PeriodExistence":
        stl_semantics = f"F[{interval_list[0][0]},{interval_list[0][1]}](G[{interval_list[1][0]},{interval_list[1][1]}] {param_list[0]})"

      elif template_type == "Response":
        stl_semantics = f"G[{interval_list[0][0]},{interval_list[0][1]}]({param_list[0]} -> F[{interval_list[1][0]},{interval_list[1][1]}]({param_list[1]}))"

      elif template_type == "RespondedExistence":
        stl_semantics = f"F[{interval_list[0][0]},{interval_list[0][1]}]({param_list[0]}) -> F[{interval_list[1][0]},{interval_list[1][1]}] ({param_list[1]})"

      elif template_type == "CoExistence":
        stl_semantics = f"F[{interval_list[0][0]}, {interval_list[0][1]}]({param_list[0]}) <-> F[{interval_list[0][0]},{interval_list[0][1]}] ({param_list[1]})"

      elif template_type == "Choice":
        stl_semantics = f"F[{interval_list[0][0]}, {interval_list[0][1]}]({param_list[0]}) v F[{interval_list[0][0]},{interval_list[0][1]}] ({param_list[1]})"

      elif template_type == "Absence":
        stl_semantics = f"¬F[{interval_list[0][0]}, {interval_list[0][1]}]({param_list[0]})"

      elif template_type == "NotPeriodExistence":
        stl_semantics = f"¬F[{interval_list[0][0]}, {interval_list[0][1]}](G[{interval_list[1][0]},{interval_list[1][1]}] {param_list[0]})"

      elif template_type == "NotResponse":
        stl_semantics = f"G[{interval_list[0][0]}, {interval_list[0][1]}]({param_list[0]} -> ¬F[{interval_list[1][0]},{interval_list[1][1]}]({param_list[1]}))"

      elif template_type == "NotCoExistence":
        stl_semantics = f"¬(F[{interval_list[0][0]}, {interval_list[0][1]}]({param_list[0]}) ∧ F[{interval_list[0][0]},{interval_list[0][1]}] ({param_list[1]}))"

      else:
        raise ValueError(f"Unknown template type: {template_type}")


      return stl_semantics
