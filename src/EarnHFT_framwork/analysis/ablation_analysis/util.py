import numpy as np

def calculate_holding_position_time(action_list):
    """Tinh toan thoi gian giu vi the trung binh"""
    holding_period_list = []
    previous_action = 0
    holding_period = 1
    
    for action in action_list:
        if action != previous_action:
            if previous_action != 0:
                holding_period_list.append(holding_period)
            previous_action = action
            holding_period = 1
        else:
            holding_period += 1
            
    if previous_action != 0:
        holding_period_list.append(holding_period)
        
    if len(holding_period_list) == 0:
        return 0.0
            
    return np.mean(holding_period_list)
