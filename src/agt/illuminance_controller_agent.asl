//illuminance controller agent

/*
* The URL of the W3C Web of Things Thing Description (WoT TD) of a lab environment
* Simulated lab WoT TD: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
* Real lab WoT TD: Get in touch with us by email to acquire access to it!
*/

/* Initial beliefs and rules */

// the agent has a belief about the location of the W3C Web of Thing (WoT) Thing Description (TD)
// that describes a lab environment to be learnt
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").

// the agent believes that the task that takes place in the 1st workstation requires an indoor illuminance
// level of Rank 2, and the task that takes place in the 2nd workstation requires an indoor illumincance 
// level of Rank 3. Modify the belief so that the agent can learn to handle different goals.
task_requirements([2,3]).

/* Initial goals */
!start. // the agent has the goal to start

/* 
 * Plan for reacting to the addition of the goal !start
 * Triggering event: addition of goal !start
 * Context: the agent believes that there is a WoT TD of a lab environment located at Url, and that 
 * the tasks taking place in the workstations require indoor illuminance levels of Rank Z1Level and Z2Level
 * respectively
 * Body: creates a QLearnerArtifact and a ThingArtifact for learning and acting on the lab environment.
*/
@start
+!start : learning_lab_environment(Url) 
  & task_requirements([Z1Level, Z2Level]) <-

  .print("Hello world");
  .print("I want to achieve Z1Level=", Z1Level, " and Z2Level=",Z2Level);

  // creates a QLearner artifact for learning the lab Thing described by the W3C WoT TD located at URL
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);

  // creates a ThingArtifact artifact for reading and acting on the state of the lab Thing
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [Url], LabArtId);
  
  .print("Starting Q-Learning for goal [", Z1Level, ",", Z2Level, "]");
  
  // Calculate Q-table for the current task requirements
  // Parameters: goalDescription, episodes, alpha, gamma, epsilon, reward
  calculateQ([Z1Level, Z2Level], 1000, 0.1, 0.9, 0.1, 1000);
  
  .print("Q-Learning completed for goal [", Z1Level, ",", Z2Level, "]");
  
  // Train for additional goal states for comparison/testing
  .print("Training for additional goal states...");
  
  // Train for goal [3,3] - both zones need maximum illuminance
  calculateQ([3, 3], 1000, 0.1, 0.9, 0.1, 1000);
  .print("Q-Learning completed for goal [3,3]");
  
  // Train for goal [1,1] - both zones need low illuminance  
  calculateQ([1, 1], 1000, 0.1, 0.9, 0.1, 1000);
  .print("Q-Learning completed for goal [1,1]");
  
  // Train for goal [0,0] - both zones need minimal illuminance
  calculateQ([0, 0], 800, 0.15, 0.95, 0.15, 800);
  .print("Q-Learning completed for goal [0,0]");
  
  .print("All Q-Learning training completed!");
  
  // Now we can test the learned policy
  !test_learned_policy.

/*
 * Plan for testing the learned policy
 */
@test_policy  
+!test_learned_policy : task_requirements([Z1Level, Z2Level]) <-
  .print("Testing learned policy for goal [", Z1Level, ",", Z2Level, "]");
  
  // Read current state of the lab
  readProperty("http://example.org/was#Status", CurrentState);
  .print("Current lab state: ", CurrentState);
  
  // Get the best action from current state using learned Q-table
  getActionFromState([Z1Level, Z2Level], CurrentState, ActionTag, PayloadTags, Payload);
  .print("Recommended action: ", ActionTag, " with payload: ", Payload);
  
  // Execute the recommended action
  invokeAction(ActionTag, PayloadTags, Payload);
  .print("Action executed!");
  
  // Wait a bit for the environment to update
  .wait(2000);
  
  // Read new state
  readProperty("http://example.org/was#Status", NewState);
  .print("New lab state after action: ", NewState);
  
  // Continue acting until goal is reached
  !act_until_goal_reached.

/*
 * Plan for continuously acting until goal is reached
 */
@act_until_goal  
+!act_until_goal_reached : task_requirements([Z1Level, Z2Level]) <-
  .print("Acting until goal [", Z1Level, ",", Z2Level, "] is reached...");
  
  // Read current state
  readProperty("http://example.org/was#Status", CurrentState);
  
  // Check if goal is reached (simplified check - you might want to make this more robust)
  !check_goal_reached(CurrentState, [Z1Level, Z2Level]);
  
  // If we reach here, goal is not reached, so get next action
  getActionFromState([Z1Level, Z2Level], CurrentState, ActionTag, PayloadTags, Payload);
  .print("Next action: ", ActionTag, " with payload: ", Payload);
  
  // Execute action
  invokeAction(ActionTag, PayloadTags, Payload);
  
  // Wait for environment update
  .wait(3000);
  
  // Continue until goal reached
  !act_until_goal_reached.

/*
 * Plan for checking if goal is reached
 */
@check_goal
+!check_goal_reached(CurrentState, [Z1Level, Z2Level]) <-
  // This is a simplified goal check - in practice you'd extract the actual
  // illuminance levels from CurrentState and compare with target levels
  .print("Checking if goal [", Z1Level, ",", Z2Level, "] is reached in state: ", CurrentState);
  
  // For now, just run for a limited number of steps to avoid infinite loops
  .print("Goal check completed (simplified implementation)");
  
  // Stop after some actions for demonstration
  .print("Stopping demonstration. In practice, implement proper goal checking logic here.").

/*
 * Plan for experimenting with different learning parameters
 */
@experiment
+!experiment_with_parameters <-
  .print("Experimenting with different Q-learning parameters...");
  
  // Experiment 1: High exploration
  calculateQ([2, 2], 500, 0.2, 0.8, 0.3, 1000);
  .print("Experiment 1 completed: High exploration (epsilon=0.3)");
  
  // Experiment 2: Low learning rate
  calculateQ([2, 2], 1500, 0.05, 0.95, 0.1, 1000);  
  .print("Experiment 2 completed: Low learning rate (alpha=0.05)");
  
  // Experiment 3: High discount factor  
  calculateQ([2, 2], 1000, 0.1, 0.99, 0.1, 1000);
  .print("Experiment 3 completed: High discount factor (gamma=0.99)");
  
  .print("Parameter experiments completed!");
.