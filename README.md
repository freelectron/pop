# Smart Charging Agent


## Aim 

Creating a robust policy with recent learning and planning techniques.

## Use case

Stationary Battery charging

## Deliverables 
    
   - Configurable simulators for a stationary battery
   
      * can be extended to the class CHP's (Combined Heat-Power)
    
   - Better forecasting methods imbalance markets
    
      * new ML forecasting model based on LSTM
      * statistical uncertainty estimates
    
   - Benchmark policy
      
      * rule-based (RB) policy: experiments with GC policy and making a policy dependent on SoC 
      
   - Extended policy
    
      * iteration 1 |#deepRL #model-free #neuralnets| : Deep Q-learning agent able to peform non-linear optimization
      based on solely on the PTU_0 bucket probabilities


## Results & insights 

 - We can trade more efficeintly with deep RL methods. A well-configured deep RL agent can make (dis)-charge
 decisions with a better ration of __revenue/energy_steered__ than an optimized RB policy.

## Further Research

 - Extend the simulator to Combined Heat-Power CHP's units
  
 - Incorporating of strict contraints in RL 
 
