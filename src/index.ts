// Load environment variables
import 'dotenv/config';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { McpAgent, McpAgentOptions } from './agents/McpAgent';
import * as fs from 'fs';
import * as path from 'path';

// Export agent classes and interfaces
export { McpAgent, McpAgentOptions } from './agents/McpAgent';

// Initialize Google AI provider
const google = createGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY || '',
});

// Read MCP configuration
const getMcpConfig = (): { mcpServers: Record<string, any> } => {
  try {
    const configPath = path.resolve(process.cwd(), 'mcp-config.json');
    const configContent = fs.readFileSync(configPath, 'utf-8');
    return JSON.parse(configContent);
  } catch (error) {
    console.error('Error reading MCP configuration:', error);
    return { mcpServers: {} };
  }
};

/**
 * Creates a Brave Search MCP agent using Gemini Pro 2.5
 * @returns The configured MCP agent
 */
export const createBraveSearchAgent = async () => {
  // Get MCP configuration
  const mcpConfig = getMcpConfig();
  const braveSearchConfig = mcpConfig.mcpServers['brave-search'];
  
  // Create agent options
  const agentOptions: McpAgentOptions = {
    name: 'brave-search-agent',
    version: '1.0.0',
    model: 'gemini-2.5-pro-preview-05-06',
    provider: google,
    transportType: 'stdio',
    stdioParams: {
      command: process.execPath,  // Path to the current node executable
      args: ["./node_modules/@modelcontextprotocol/server-brave-search/dist/index.js"],
      env: {
        BRAVE_API_KEY: process.env.BRAVE_SEARCH_API_KEY || '',
        PATH: process.env.PATH || ''  // Pass the current PATH with fallback
      },
    },
    system: 'You are a helpful assistant that can search the web for information.',
    temperature: 0.7,
    maxTokens: 1024,
  };
  
  // Create MCP agent with Gemini model
  const agent = new McpAgent(agentOptions);

  // Connect to the MCP server
  await agent.connect();
  console.log('Connected to Brave Search MCP server');
  
  return agent;
};

/**
 * Test function to demonstrate the Brave Search MCP agent
 * @param query The search query to process
 */
export const testBraveSearchAgent = async (query: string) => {
  try {
    console.log(`Testing Brave Search agent with query: "${query}"`);
    
    // Create and connect the agent
    const agent = await createBraveSearchAgent();
    
    // Process the query
    const result = await agent.process(query);
    
    console.log('\nSearch Results:');
    console.log('==============');
    console.log(result.finalAnswer);
    
    if (result.toolCalls && result.toolCalls.length > 0) {
      console.log('\nTool Calls:');
      console.log('===========');
      result.toolCalls.forEach((call, index) => {
        console.log(`Tool Call ${index + 1}: ${call.toolName}`);
        console.log(`Arguments: ${JSON.stringify(call.args, null, 2)}`);
      });
    }
    
    return result;
  } catch (error) {
    console.error('Error testing Brave Search agent:', error);
    throw error;
  }
};

// Example usage (uncomment to run)
/*
if (require.main === module) {
  testBraveSearchAgent('What are the latest developments in quantum computing?')
    .then(() => console.log('Test completed'))
    .catch(error => console.error('Test failed:', error));
}
*/
