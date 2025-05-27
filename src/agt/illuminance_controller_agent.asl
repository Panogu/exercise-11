//illuminance controller agent - Clean Version with Q-Table Export

/* Initial beliefs and rules */
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").
task_requirements([2,3]).

/* Initial goals */
!start.

/* 
 * Main startup plan - Complete Task 2 Implementation
 */
+!start : learning_lab_environment(Url) 
  & task_requirements([Z1Level, Z2Level]) <-

  .print("=== Illuminance Controller Agent Starting ===");
  .print("Target: Z1Level=", Z1Level, " Z2Level=", Z2Level);

  /* Create the QLearner artifact */
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);
  
  /* Create the ThingArtifact for lab interaction */
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [Url], LabArtId);
  
  .print("=== Starting Q-Learning Training ===");
  
  /* Train Q-Learning with improved parameters for better exploration */
  /* Parameters: goal, episodes, alpha, gamma, epsilon, reward */
  calculateQ([Z1Level, Z2Level], 100, 0.2, 0.9, 0.3, 1000);
  .print("Primary Q-Learning completed for goal [", Z1Level, ",", Z2Level, "]");
  
  /* Train for additional goals for comparison */
  calculateQ([3, 3], 150, 0.2, 0.9, 0.3, 1000);
  .print("Q-Learning completed for goal [3,3]");
  
  calculateQ([1, 1], 150, 0.2, 0.9, 0.3, 1000);
  .print("Q-Learning completed for goal [1,1]");
  
  calculateQ([0, 0], 120, 0.25, 0.9, 0.35, 800);
  .print("Q-Learning completed for goal [0,0]");
  
  .print("=== All Q-Learning Training Completed ===");
  
  /* Export Q-tables for submission */
  .print("=== Exporting Q-Tables for Submission ===");
  exportAllQTables;
  .print("Q-tables exported to files");
  
  /* Print Q-tables to console for report */
  .print("=== Q-TABLE FOR PRIMARY GOAL [", Z1Level, ",", Z2Level, "] ===");
  printQTableForReport([Z1Level, Z2Level]);
  
  .print("=== Q-TABLE FOR GOAL [3,3] ===");
  printQTableForReport([3, 3]);
  
  .print("=== Q-TABLE FOR GOAL [1,1] ===");
  printQTableForReport([1, 1]);
  
  .print("=== Q-TABLE FOR GOAL [0,0] ===");
  printQTableForReport([0, 0]);
  
  /* Get and display statistics */
  getQTableStats([Z1Level, Z2Level], PrimaryStats);
  .print("Statistics for primary goal: ", PrimaryStats);
  
  getQTableStats([3, 3], Stats33);
  .print("Statistics for goal [3,3]: ", Stats33);
  
  getQTableStats([1, 1], Stats11);
  .print("Statistics for goal [1,1]: ", Stats11);
  
  getQTableStats([0, 0], Stats00);
  .print("Statistics for goal [0,0]: ", Stats00);
  
  .print("=== Training and Export Complete - Starting Goal Achievement ===");
  
  /* Show initial state before acting */
  !get_current_state_info;
  
  /* Start acting to achieve the goal */
  !achieve_goal.

/*
 * Helper plan to get and display current state information
 */
+!get_current_state_info <-
  .print("Reading current lab state...");
  readCurrentLabState(CurrentState);
  .print("Current state: ", CurrentState);
  getStateDescription(StateDesc);
  .print("State description: ", StateDesc).

/*
 * Main goal achievement plan
 */
+!achieve_goal : task_requirements([Z1Level, Z2Level]) <-
  .print("Attempting to reach goal [", Z1Level, ",", Z2Level, "]");
  !act_towards_goal(0).

/*
 * Recursive acting plan with step counter
 */
+!act_towards_goal(Step) : Step < 15 <-
  .print("--- Action Step ", Step, " ---");
  
  /* Read current state using QLearner's capability */
  readCurrentLabState(CurrentState);
  .print("Current lab state: ", CurrentState);
  
  /* Get best action using learned Q-table */
  task_requirements([Z1Level, Z2Level]);
  getActionFromState([Z1Level, Z2Level], CurrentState, ActionTag, PayloadTags, Payload);
  .print("Selected action: ", ActionTag, " with payload: ", Payload);
  
  /* Execute the action */
  invokeAction(ActionTag, PayloadTags, Payload);
  .print("Action executed successfully");
  
  /* Wait for environment to update */
  .wait(3000);
  
  /* Continue to next step */
  NextStep = Step + 1;
  !act_towards_goal(NextStep).

/*
 * Plan when maximum steps reached
 */
+!act_towards_goal(Step) : Step >= 15 <-
  .print("=== Reached maximum steps (15) ===");
  readCurrentLabState(FinalState);
  .print("Final achieved state: ", FinalState);
  getStateDescription(FinalStateDesc);
  .print("Final state description: ", FinalStateDesc);
  task_requirements([Z1Level, Z2Level]);
  .print("Target was: [Z1Level=", Z1Level, ", Z2Level=", Z2Level, "]");
  .print("=== Goal Achievement Attempt Complete ===").

/*
 * Error handling plan
 */
-!achieve_goal <-
  .print("ERROR: Failed to achieve goal");
  readCurrentLabState(ErrorState);
  .print("State when error occurred: ", ErrorState).

/*
 * Plan for manually exporting Q-tables
 */
+!export_qtables_manual <-
  .print("=== Manual Q-Table Export ===");
  exportAllQTables;
  task_requirements([Z1, Z2]);
  exportQTableForGoal([Z1, Z2], "primary_goal_qtable");
  exportQTableForGoal([3, 3], "max_illuminance_qtable"); 
  exportQTableForGoal([1, 1], "low_illuminance_qtable");
  exportQTableForGoal([0, 0], "min_illuminance_qtable");
  .print("Manual export completed").

/*
 * Plan for testing the learned policy
 */
+!test_policy : task_requirements([Z1Level, Z2Level]) <-
  .print("=== Testing Learned Policy ===");
  readCurrentLabState(TestState);
  .print("Current state for testing: ", TestState);
  
  getActionFromState([Z1Level, Z2Level], TestState, TestAction, TestTags, TestPayload);
  .print("Recommended action: ", TestAction, " payload: ", TestPayload);
  
  /* Test other goals too */
  getActionFromState([3, 3], TestState, Action33, Tags33, Payload33);
  .print("For goal [3,3] would recommend: ", Action33, " payload: ", Payload33);
  
  getActionFromState([1, 1], TestState, Action11, Tags11, Payload11);
  .print("For goal [1,1] would recommend: ", Action11, " payload: ", Payload11);
  
  getActionFromState([0, 0], TestState, Action00, Tags00, Payload00);
  .print("For goal [0,0] would recommend: ", Action00, " payload: ", Payload00);
  
  .print("Policy testing completed").

/*
 * Plan for demonstrating different goals sequentially
 */
+!demo_all_goals <-
  .print("=== Demonstrating All Learned Goals ===");
  
  .print("Testing goal [3,3] - Maximum illuminance");
  !single_goal_demo([3, 3], 8);
  
  .wait(5000);
  
  .print("Testing goal [1,1] - Low illuminance");  
  !single_goal_demo([1, 1], 8);
  
  .wait(5000);
  
  .print("Testing goal [0,0] - Minimal illuminance");
  !single_goal_demo([0, 0], 8);
  
  .print("=== All goal demonstrations complete ===").

/*
 * Helper plan for demonstrating a single goal
 */
+!single_goal_demo(Goal, MaxSteps) <-
  .print("Demonstrating goal: ", Goal);
  !demo_goal_steps(Goal, 0, MaxSteps).

+!demo_goal_steps(Goal, Step, MaxSteps) : Step < MaxSteps <-
  .print("Demo step ", Step, " for goal ", Goal);
  readCurrentLabState(State);
  getActionFromState(Goal, State, Action, Tags, Payload);
  .print("Action: ", Action, " with payload: ", Payload);
  
  invokeAction(Action, Tags, Payload);
  .wait(2000);
  
  NextStep = Step + 1;
  !demo_goal_steps(Goal, NextStep, MaxSteps).

+!demo_goal_steps(Goal, Step, MaxSteps) : Step >= MaxSteps <-
  .print("Demo completed for goal ", Goal);
  readCurrentLabState(FinalState);
  getStateDescription(FinalDesc);
  .print("Final state: ", FinalDesc).

/*
 * Plan for analyzing Q-table quality
 */
+!analyze_learning_quality <-
  .print("=== Q-Table Quality Analysis ===");
  
  task_requirements([Z1, Z2]);
  getQTableStats([Z1, Z2], Stats1);
  getQTableStats([3, 3], Stats2);
  getQTableStats([1, 1], Stats3);
  getQTableStats([0, 0], Stats4);
  
  .print("Analysis completed - check statistics above").

/*
 * Utility plan for quick state check
 */
+!check_state <-
  readCurrentLabState(State);
  getStateDescription(Desc);
  .print("Quick state check - State: ", State);
  .print("Description: ", Desc).