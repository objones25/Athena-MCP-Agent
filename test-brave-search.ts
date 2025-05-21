import { testBraveSearchAgent } from './src/index';

// Define a test query
const testQuery = 'What are the latest advancements in artificial intelligence?';

// Run the test
console.log('Starting Brave Search MCP Agent test with Gemini Pro 2.5...');
testBraveSearchAgent(testQuery)
  .then((result) => {
    console.log('\nTest completed successfully!');
    console.log('Full result object:', JSON.stringify(result, null, 2));
  })
  .catch((error) => {
    console.error('Test failed:', error);
  });
