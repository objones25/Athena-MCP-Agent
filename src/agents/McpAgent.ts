import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { generateText, tool } from 'ai';
import { z } from 'zod';
import { Tool } from '@modelcontextprotocol/sdk/types.js';

/**
 * Options for configuring an MCP Agent
 */
export interface McpAgentOptions {
  /** Name of the agent */
  name: string;
  
  /** Version of the agent */
  version: string;
  
  /** Model to use for LLM calls (e.g., 'gpt-4.1') */
  model: string;
  
  /** Provider function (e.g., openai, anthropic) */
  provider?: any;
  
  /** Capabilities to advertise to the MCP server */
  capabilities?: any;
  
  /** URL of the MCP server (for HTTP/SSE transports) */
  serverUrl?: URL;
  
  /** Transport type to use ('streamable-http', 'sse', or 'stdio') */
  transportType?: 'streamable-http' | 'sse' | 'stdio';
  
  /** Parameters for stdio transport (required when transportType is 'stdio') */
  stdioParams?: {
    /** The executable to run to start the server */
    command: string;
    /** Command line arguments to pass to the executable */
    args?: string[];
    /** Environment variables to pass to the process */
    env?: Record<string, string>;
    /** Working directory for the process */
    cwd?: string;
  };
  
  /** Whether to automatically load all capabilities after connecting */
  autoLoadCapabilities?: boolean;
  
  // Vercel AI SDK generateText parameters
  
  /** System message to include in prompts */
  system?: string;
  
  /** Maximum number of tokens to generate */
  maxTokens?: number;
  
  /** Temperature setting (0-1). Controls randomness. */
  temperature?: number;
  
  /** Nucleus sampling (0-1). Alternative to temperature. */
  topP?: number;
  
  /** Only sample from the top K options for each token */
  topK?: number;
  
  /** Presence penalty. Affects likelihood of repeating information from prompt. */
  presencePenalty?: number;
  
  /** Frequency penalty. Affects likelihood of repeating the same words. */
  frequencyPenalty?: number;
  
  /** Stop sequences. Model will stop generating when these are generated. */
  stopSequences?: string[];
  
  /** Seed for random sampling. Enables deterministic results if supported. */
  seed?: number;
  
  /** Maximum number of retries for API calls */
  maxRetries?: number;
  
  /** AbortSignal to cancel requests */
  abortSignal?: AbortSignal;
  
  /** Additional HTTP headers for requests */
  headers?: Record<string, string>;
  
  /** Maximum number of sequential LLM calls (steps) */
  maxSteps?: number;
  
  /** Generate unique IDs for messages */
  experimental_generateMessageId?: () => string;
  
  /** Continue generating if finish reason is "length" */
  experimental_continueSteps?: boolean;
  
  /** Telemetry settings */
  experimental_telemetry?: any;
  
  /** Provider-specific options */
  providerOptions?: any;
  
  /** Function to prepare different settings for each step */
  experimental_prepareStep?: (options: {
    steps: any[];
    stepNumber: number;
    maxSteps: number;
    model: any;
  }) => Promise<any>;
  
  /** Function to repair tool calls that fail to parse */
  experimental_repairToolCall?: any;
  
  /** Callback for when each step finishes */
  onStepFinish?: (stepResult: any) => Promise<void> | void;
  
  /** Whether to include resources in the prompt context */
  includeResources?: boolean;
}

/**
 * MCP Agent that combines the MCP Client with the Vercel AI SDK
 * Automatically loads and registers all capabilities from the server
 */
export class McpAgent {
  /** The MCP client instance */
  protected client: Client;
  
  /** Configuration options */
  protected options: McpAgentOptions;
  
  /** Tools available to the LLM */
  protected tools: Record<string, any> = {};
  
  /** Resources loaded from the server */
  protected resources: Record<string, any> = {};
  
  /** Prompts loaded from the server */
  protected prompts: Record<string, any> = {};
  
  /**
   * Creates a new MCP Agent
   * @param options Configuration options
   */
  constructor(options: McpAgentOptions) {
    this.options = {
      maxSteps: 5,
      transportType: 'streamable-http',
      autoLoadCapabilities: true,
      ...options
    };
    
    // Initialize MCP Client
    this.client = new Client({
      name: options.name,
      version: options.version
    }, {
      capabilities: options.capabilities
    });
  }
  
  /**
   * Connects to the MCP server and optionally loads all capabilities
   * @returns The agent instance for chaining
   */
  async connect() {
    try {
      let transport;
      
      switch (this.options.transportType) {
        case 'streamable-http':
          if (!this.options.serverUrl) {
            throw new Error('serverUrl is required for streamable-http transport');
          }
          transport = new StreamableHTTPClientTransport(this.options.serverUrl);
          break;
        case 'stdio':
          if (!this.options.stdioParams) {
            throw new Error('stdioParams is required for stdio transport');
          }
          transport = new StdioClientTransport({
            command: this.options.stdioParams.command,
            args: this.options.stdioParams.args,
            env: this.options.stdioParams.env,
            cwd: this.options.stdioParams.cwd
          });
          break;
        case 'sse':
        default:
          if (!this.options.serverUrl) {
            throw new Error('serverUrl is required for sse transport');
          }
          transport = new SSEClientTransport(this.options.serverUrl);
          break;
      }
      
      await this.client.connect(transport);
      console.log(`Connected to MCP server using ${this.options.transportType} transport`);
      
      // Automatically load all capabilities if enabled
      if (this.options.autoLoadCapabilities) {
        await this.loadAllCapabilities();
      }
      
      return this;
    } catch (error) {
      console.error('Failed to connect to MCP server:', error);
      throw error;
    }
  }
  
  /**
   * Loads all capabilities from the server
   * @returns The agent instance for chaining
   */
  async loadAllCapabilities() {
    try {
      // Load all capabilities in parallel
      await Promise.all([
        this.loadTools(),
        this.loadResources(),
        this.loadPrompts()
      ]);
      
      return this;
    } catch (error) {
      console.error('Error loading capabilities:', error);
      throw error;
    }
  }
  
  /**
   * Loads and registers all tools from the server
   * @returns The agent instance for chaining
   */
  async loadTools() {
    try {
      const serverCapabilities = this.client.getServerCapabilities();
      
      // Check if the server supports tools
      if (!serverCapabilities?.tools) {
        console.log('Server does not support tools');
        return this;
      }
      
      const { tools } = await this.client.listTools();
      console.log(`Loaded ${tools.length} tools from server`);
      
      // Register each tool
      for (const tool of tools) {
        this.registerMcpTool(tool);
      }
      
      return this;
    } catch (error) {
      console.error('Error loading tools:', error);
      return this;
    }
  }
  
  /**
   * Loads all resources from the server
   * @returns The agent instance for chaining
   */
  async loadResources() {
    try {
      const serverCapabilities = this.client.getServerCapabilities();
      
      // Check if the server supports resources
      if (!serverCapabilities?.resources) {
        console.log('Server does not support resources');
        return this;
      }
      
      const { resources } = await this.client.listResources();
      console.log(`Found ${resources.length} resources on server`);
      
      // Load each resource
      for (const resource of resources) {
        try {
          const result = await this.client.readResource({
            uri: resource.uri
          });
          
          this.resources[resource.name] = result.contents;
        } catch (error) {
          console.error(`Failed to load resource ${resource.name}:`, error);
        }
      }
      
      return this;
    } catch (error) {
      console.error('Error loading resources:', error);
      return this;
    }
  }
  
  /**
   * Loads all prompts from the server
   * @returns The agent instance for chaining
   */
  async loadPrompts() {
    try {
      const serverCapabilities = this.client.getServerCapabilities();
      
      // Check if the server supports prompts
      if (!serverCapabilities?.prompts) {
        console.log('Server does not support prompts');
        return this;
      }
      
      const { prompts } = await this.client.listPrompts();
      console.log(`Loaded ${prompts.length} prompts from server`);
      
      // Store prompts for later use
      for (const prompt of prompts) {
        this.prompts[prompt.name] = prompt;
      }
      
      return this;
    } catch (error) {
      console.error('Error loading prompts:', error);
      return this;
    }
  }
  
  /**
   * Registers an MCP tool as a Vercel AI tool
   * @param mcpTool The MCP tool to register
   * @returns The agent instance for chaining
   */
  protected registerMcpTool(mcpTool: Tool) {
    try {
      // Create a Zod schema from the tool's input schema
      const paramSchema = this.createZodSchemaFromJsonSchema(mcpTool.inputSchema);
      
      // Register the tool
      this.registerTool(
        mcpTool.name,
        mcpTool.description || `MCP Tool: ${mcpTool.name}`,
        paramSchema,
        async (args: any) => {
          try {
            const result = await this.client.callTool({
              name: mcpTool.name,
              arguments: args
            });
            
            return result.structuredContent || result.content;
          } catch (error) {
            console.error(`Error calling MCP tool ${mcpTool.name}:`, error);
            return { error: `Failed to call MCP tool ${mcpTool.name}: ${error}` };
          }
        }
      );
      
      return this;
    } catch (error) {
      console.error(`Error registering MCP tool ${mcpTool.name}:`, error);
      return this;
    }
  }
  
  /**
   * Helper to convert JSON Schema to Zod schema
   * @param schema JSON Schema object
   * @returns Zod schema
   */
  protected createZodSchemaFromJsonSchema(schema: any): z.ZodObject<any> {
    // Handle missing schema
    if (!schema || !schema.properties) {
      return z.object({});
    }
    
    const properties: Record<string, any> = {};
    
    // Process each property in the schema
    for (const [key, prop] of Object.entries(schema.properties)) {
      const propDef = prop as any;
      let zodProp;
      
      // Convert JSON Schema types to Zod types
      switch (propDef.type) {
        case 'string':
          zodProp = z.string();
          
          // Handle string formats
          if (propDef.format === 'date-time') {
            zodProp = z.string().datetime();
          } else if (propDef.format === 'email') {
            zodProp = z.string().email();
          } else if (propDef.format === 'uri') {
            zodProp = z.string().url();
          }
          
          // Handle enums
          if (propDef.enum && Array.isArray(propDef.enum)) {
            zodProp = z.enum(propDef.enum as [string, ...string[]]);
          }
          break;
          
        case 'number':
        case 'integer':
          zodProp = z.number();
          
          // Handle numeric constraints
          if (propDef.minimum !== undefined) {
            zodProp = zodProp.min(propDef.minimum);
          }
          if (propDef.maximum !== undefined) {
            zodProp = zodProp.max(propDef.maximum);
          }
          break;
          
        case 'boolean':
          zodProp = z.boolean();
          break;
          
        case 'object':
          zodProp = this.createZodSchemaFromJsonSchema(propDef);
          break;
          
        case 'array':
          // Handle array items
          if (propDef.items) {
            if (propDef.items.type === 'string') {
              zodProp = z.array(z.string());
            } else if (propDef.items.type === 'number' || propDef.items.type === 'integer') {
              zodProp = z.array(z.number());
            } else if (propDef.items.type === 'boolean') {
              zodProp = z.array(z.boolean());
            } else if (propDef.items.type === 'object') {
              zodProp = z.array(this.createZodSchemaFromJsonSchema(propDef.items));
            } else {
              zodProp = z.array(z.any());
            }
          } else {
            zodProp = z.array(z.any());
          }
          break;
          
        default:
          zodProp = z.any();
      }
      
      // Add description if available
      if (propDef.description) {
        zodProp = zodProp.describe(propDef.description);
      }
      
      properties[key] = zodProp;
    }
    
    let zodSchema = z.object(properties);
    
    // Handle required properties
    if (schema.required && Array.isArray(schema.required)) {
      // Create a new properties object with required fields properly marked
      const requiredProperties: Record<string, any> = {};
      
      for (const [key, prop] of Object.entries(properties)) {
        if (schema.required.includes(key)) {
          // For required properties, we need to ensure they're not optional
          requiredProperties[key] = prop;
        } else {
          // For optional properties, we need to make them optional
          requiredProperties[key] = prop.optional();
        }
      }
      
      // Create a new schema with the updated properties
      zodSchema = z.object(requiredProperties);
    }
    
    return zodSchema;
  }
  
  /**
   * Registers a tool that will be available to the LLM
   * @param name Name of the tool
   * @param description Description of the tool
   * @param parameters Zod schema for the tool parameters
   * @param execute Function to execute when the tool is called
   * @returns The agent instance for chaining
   */
  registerTool(name: string, description: string, parameters: z.ZodObject<any>, execute: Function) {
    this.tools[name] = tool({
      description,
      parameters,
      execute: execute as any
    });
    
    return this;
  }
  
  /**
   * Processes a user prompt using the LLM and available tools
   * @param prompt The user prompt to process
   * @param options Additional options for processing
   * @returns The result of processing the prompt
   */
  async process(prompt: string, options: { includeResources?: boolean } = {}) {
    try {
      // Include resources in the prompt if available and not explicitly disabled
      const includeResources = options.includeResources !== false && 
        this.options.includeResources !== false && 
        Object.keys(this.resources).length > 0;
      
      let enhancedPrompt = prompt;
      
      if (includeResources) {
        enhancedPrompt = `
Context Information:
${JSON.stringify(this.resources, null, 2)}

User Query:
${prompt}
`;
      }
      
      // Call generateText with the appropriate parameters
      const result = await generateText({
        model: this.getModelProvider(),
        prompt: enhancedPrompt,
        tools: this.tools,
        
        // Pass through all Vercel AI SDK parameters from options
        ...(this.options.system !== undefined && { system: this.options.system }),
        ...(this.options.maxTokens !== undefined && { maxTokens: this.options.maxTokens }),
        ...(this.options.temperature !== undefined && { temperature: this.options.temperature }),
        ...(this.options.topP !== undefined && { topP: this.options.topP }),
        ...(this.options.topK !== undefined && { topK: this.options.topK }),
        ...(this.options.presencePenalty !== undefined && { presencePenalty: this.options.presencePenalty }),
        ...(this.options.frequencyPenalty !== undefined && { frequencyPenalty: this.options.frequencyPenalty }),
        ...(this.options.stopSequences !== undefined && { stopSequences: this.options.stopSequences }),
        ...(this.options.seed !== undefined && { seed: this.options.seed }),
        ...(this.options.maxRetries !== undefined && { maxRetries: this.options.maxRetries }),
        ...(this.options.abortSignal !== undefined && { abortSignal: this.options.abortSignal }),
        ...(this.options.headers !== undefined && { headers: this.options.headers }),
        ...(this.options.maxSteps !== undefined && { maxSteps: this.options.maxSteps }),
        ...(this.options.experimental_generateMessageId !== undefined && { 
          experimental_generateMessageId: this.options.experimental_generateMessageId 
        }),
        ...(this.options.experimental_continueSteps !== undefined && { 
          experimental_continueSteps: this.options.experimental_continueSteps 
        }),
        ...(this.options.experimental_telemetry !== undefined && { 
          experimental_telemetry: this.options.experimental_telemetry 
        }),
        ...(this.options.providerOptions !== undefined && { 
          providerOptions: this.options.providerOptions 
        }),
        ...(this.options.experimental_prepareStep !== undefined && { 
          experimental_prepareStep: this.options.experimental_prepareStep 
        }),
        ...(this.options.experimental_repairToolCall !== undefined && { 
          experimental_repairToolCall: this.options.experimental_repairToolCall 
        }),
        ...(this.options.onStepFinish !== undefined && { 
          onStepFinish: this.options.onStepFinish 
        })
      });
      
      return {
        steps: result.steps,
        finalAnswer: result.text,
        toolCalls: result.toolCalls,
        toolResults: result.toolResults,
        reasoning: result.reasoning,
        usage: result.usage,
        finishReason: result.finishReason
      };
    } catch (error) {
      console.error('Error processing prompt:', error);
      throw error;
    }
  }
  
  /**
   * Gets the model provider based on options
   * @returns The configured model provider
   */
  protected getModelProvider() {
    // Use the specified provider or return a simple model identifier
    if (this.options.provider) {
      return this.options.provider(this.options.model);
    }
    
    // If no provider is specified, just return the model name
    // This is a fallback and would need to be properly handled in a real implementation
    return this.options.model;
  }
  
  /**
   * Gets the server capabilities
   * @returns The server capabilities
   */
  getServerCapabilities() {
    return this.client.getServerCapabilities();
  }
  
  /**
   * Gets the server version
   * @returns The server version
   */
  getServerVersion() {
    return this.client.getServerVersion();
  }
  
  /**
   * Gets all loaded resources
   * @returns The loaded resources
   */
  getResources() {
    return { ...this.resources };
  }
  
  /**
   * Gets all loaded prompts
   * @returns The loaded prompts
   */
  getPrompts() {
    return { ...this.prompts };
  }
  
  /**
   * Gets all registered tools
   * @returns The registered tools
   */
  getTools() {
    return { ...this.tools };
  }
}
