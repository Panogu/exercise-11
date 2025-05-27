//illuminance controller agent - Task 2 Implementation

/* Initial beliefs and rules */
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").
task_requirements([2,3]).

/* Initial goals */
!start.

/* 
 * Main startup plan - Task 2.2 Implementation
 */
+!start : learning_lab_environment(Url) 
  & task_requirements([Z1Level, Z2Level]) <-

  .print("=== Illuminance Controller Agent Starting ===");
  .print("Target: Z1Level=", Z1Level, " Z2Level=", Z2Level);

  /* Create the QLearner artifact - Task 2.2 */
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);
  
  /* Create the ThingArtifact for lab interaction */
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [Url], LabArtId);
  
  .print("=== Starting Q-Learning Training ===");
  
  /* Train Q-Learning for our goal - Task 2.2 */
  calculateQ([Z1Level, Z2Level], 1000, 0.1, 0.9, 0.1, 1000);
  .print("Primary Q-Learning completed for goal [", Z1Level, ",", Z2Level, "]");
  
  /* Train for additional goals for robustness */
  //calculateQ([3, 3], 800, 0.1, 0.9, 0.1, 1000);
  //calculateQ([1, 1], 800, 0.1, 0.9, 0.1, 1000);
  //calculateQ([0, 0], 600, 0.15, 0.9, 0.15, 800);
  .print("Additional Q-Learning training completed");
  
  .print("=== Training Complete - Starting Goal Achievement ===");
  
  /* Start acting to achieve the goal - Task 2.3 */
  !achieve_goal.

/*
 * Main goal achievement plan - Task 2.3 Implementation
 */
+!achieve_goal : task_requirements([Z1Level, Z2Level]) <-
  .print("Attempting to reach goal [", Z1Level, ",", Z2Level, "]");
  !act_towards_goal(0).

/*
 * Recursive acting plan with step counter
 */
+!act_towards_goal(Step) : Step < 15 <-
  .print("--- Action Step ", Step, " ---");
  
  /* Read current state from lab */
  readProperty("http://example.org/was#Status", CurrentState);
  .print("Current lab state: ", CurrentState);
  
  /* Get best action using learned Q-table - Task 2.3 */
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
  readProperty("http://example.org/was#Status", FinalState);
  .print("Final achieved state: ", FinalState);
  task_requirements([Z1Level, Z2Level]);
  .print("Target was: [Z1Level=", Z1Level, ", Z2Level=", Z2Level, "]");
  .print("=== Goal Achievement Attempt Complete ===").

/*
 * Error handling plan
 */
-!achieve_goal <-
  .print("ERROR: Failed to achieve goal");
  readProperty("http://example.org/was#Status", ErrorState);
  .print("State when error occurred: ", ErrorState).

/*
 * Plan for testing the learned policy manually
 */
+!test_policy : task_requirements([Z1Level, Z2Level]) <-
  .print("=== Testing Learned Policy ===");
  readProperty("http://example.org/was#Status", TestState);
  .print("Current state for testing: ", TestState);
  
  getActionFromState([Z1Level, Z2Level], TestState, TestAction, TestTags, TestPayload);
  .print("Recommended action: ", TestAction, " payload: ", TestPayload);
  
  .print("Test completed - you can manually invoke: !test_policy").

/*
 * Plan for demonstrating different goals
 */
+!demo_different_goals <-
  .print("=== Demonstrating Different Goals ===");
  
  .print("Testing goal [3,3] - Maximum illuminance");
  !single_goal_test([3, 3]);
  
  .wait(5000);
  
  .print("Testing goal [1,1] - Low illuminance");  
  !single_goal_test([1, 1]);
  
  .wait(5000);
  
  .print("Testing goal [0,0] - Minimal illuminance");
  !single_goal_test([0, 0]);
  
  .print("=== All goal demonstrations complete ===").

/*
 * Helper plan for testing a single goal
 */
+!single_goal_test(Goal) <-
  .print("Testing goal: ", Goal);
  readProperty("http://example.org/was#Status", State);
  getActionFromState(Goal, State, Action, Tags, Payload);
  .print("For goal ", Goal, " recommended action: ", Action, " with payload: ", Payload);
  
  /* Execute the action */
  invokeAction(Action, Tags, Payload);
  .wait(2000);
  
  /* Show result */
  readProperty("http://example.org/was#Status", NewState);
  .print("State after action: ", NewState).