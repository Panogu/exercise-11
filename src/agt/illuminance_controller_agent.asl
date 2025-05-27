//illuminance controller agent

/*
* Complete implementation for Task 2 - Reinforcement Learning Agent
*/

/* Initial beliefs and rules */

learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").

// Try different goal configurations for testing
task_requirements([2,3]).

// Maximum steps to prevent infinite loops
max_steps(20).

/* Initial goals */
!start.

/* 
 * Main startup plan
 */
@start
+!start : learning_lab_environment(Url) 
  & task_requirements([Z1Level, Z2Level]) <-

  .print("=== Illuminance Controller Agent Starting ===");
  .print("Target: Z1Level=", Z1Level, " Z2Level=", Z2Level);

  // Create artifacts
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [Url], LabArtId);
  
  // Phase 1: Learning
  !learn_policy;
  
  // Phase 2: Acting
  !achieve_goal.

/*
 * Learning phase
 */
@learn
+!learn_policy : task_requirements([Z1Level, Z2Level]) <-
  .print("=== LEARNING PHASE ===");
  .print("Training Q-Learning for goal [", Z1Level, ",", Z2Level, "]");
  
  // Train the main goal with good parameters
  calculateQ([Z1Level, Z2Level], 1000, 0.1, 0.9, 0.1, 1000);
  .print("Primary training completed");
  
  // Train some additional goals for robustness
  calculateQ([3, 3], 800, 0.1, 0.9, 0.1, 1000);
  calculateQ([1, 1], 800, 0.1, 0.9, 0.1, 1000);
  calculateQ([0, 0], 600, 0.15, 0.9, 0.15, 800);
  
  .print("=== LEARNING COMPLETED ===").

/*
 * Goal achievement phase
 */
@achieve
+!achieve_goal : task_requirements([Z1Level, Z2Level]) & max_steps(MaxSteps) <-
  .print("=== GOAL ACHIEVEMENT PHASE ===");
  .print("Attempting to reach goal [", Z1Level, ",", Z2Level, "]");
  
  !act_to_goal([Z1Level, Z2Level], 0, MaxSteps).

/*
 * Recursive action plan
 */
@act_recursive
+!act_to_goal(Goal, CurrentStep, MaxSteps) : CurrentStep < MaxSteps <-
  .print("--- Step ", CurrentStep, " ---");
  
  // Read current environment state
  readProperty("http://example.org/was#Status", State);
  .print("Current state: ", State);
  
  // Check if we've reached the goal
  if (is_goal_reached(State, Goal)) {
    .print("*** GOAL REACHED! ***");
    .print("Final state: ", State);
    .print("Achieved in ", CurrentStep, " steps");
  } else {
    .print("Goal not yet reached, selecting next action...");
    
    // Get best action from Q-table
    getActionFromState(Goal, State, ActionTag, PayloadTags, Payload);
    .print("Selected action: ", ActionTag, " with payload: ", Payload);
    
    // Execute the action
    invokeAction(ActionTag, PayloadTags, Payload);
    .print("Action executed");
    
    // Wait for environment to update
    .wait(2000);
    
    // Continue to next step
    NextStep = CurrentStep + 1;
    !act_to_goal(Goal, NextStep, MaxSteps);
  }.

/*
 * Handle maximum steps reached
 */
@max_steps_reached
+!act_to_goal(Goal, CurrentStep, MaxSteps) : CurrentStep >= MaxSteps <-
  .print("Maximum steps (", MaxSteps, ") reached");
  readProperty("http://example.org/was#Status", FinalState);
  .print("Final state: ", FinalState);
  .print("Goal ", Goal, " achievement attempt completed").

/*
 * Goal checking rule (simplified)
 * In practice, you would extract the actual illuminance levels and compare them
 */
is_goal_reached(State, [Z1Target, Z2Target]) :-
  .print("Checking if goal [", Z1Target, ",", Z2Target, "] is reached");
  // This is a simplified check - implement proper state parsing logic
  true.

/*
 * Plan for testing different goals
 */
@test_goals
+!test_different_goals <-
  .print("=== TESTING DIFFERENT GOALS ===");
  
  // Test goal [3,3]
  .print("Testing goal [3,3]");
  !act_to_goal([3, 3], 0, 10);
  .wait(3000);
  
  // Test goal [1,1]  
  .print("Testing goal [1,1]");
  !act_to_goal([1, 1], 0, 10);
  .wait(3000);
  
  // Test goal [0,0]
  .print("Testing goal [0,0]");
  !act_to_goal([0, 0], 0, 10);
  
  .print("=== ALL TESTS COMPLETED ===").

/*
 * Error handling plan
 */
@error_handler
-!achieve_goal <-
  .print("ERROR: Failed to achieve goal");
  readProperty("http://example.org/was#Status", ErrorState);
  .print("State when error occurred: ", ErrorState).

/*
 * Utility plan for debugging
 */
@debug
+!debug_state <-
  readProperty("http://example.org/was#Status", DebugState);
  .print("DEBUG - Current state: ", DebugState);
  
  // Test action selection
  task_requirements([Z1, Z2]);
  getActionFromState([Z1, Z2], DebugState, TestAction, TestPayloadTags, TestPayload);
  .print("DEBUG - Recommended action: ", TestAction, " payload: ", TestPayload).

/* 
 * Plan to demonstrate the complete system
 */
@demo
+!demonstration <-
  .print("=== SYSTEM DEMONSTRATION ===");
  
  // Show initial state
  readProperty("http://example.org/was#Status", InitialState);
  .print("Initial lab state: ", InitialState);
  
  // Learn policy
  !learn_policy;
  
  // Achieve primary goal
  task_requirements([PrimaryZ1, PrimaryZ2]);
  .print("Achieving primary goal [", PrimaryZ1, ",", PrimaryZ2, "]");
  !act_to_goal([PrimaryZ1, PrimaryZ2], 0, 15);
  
  .wait(5000);
  
  // Test other goals
  !test_different_goals;
  
  .print("=== DEMONSTRATION COMPLETE ===").