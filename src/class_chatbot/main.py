import uuid

from langchain_groq import ChatGroq
from typing import TypedDict, Literal

from datetime import datetime
from trustcall import create_extractor
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, SystemMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from class_chatbot.configuration import Configuration

model = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_retries=2,
)

# Inspect the tool calls made by Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "Order")
    """
    # Initialize list of changes
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                # Safely handle potential issues with 'patches'
                patches = call['args']['patches'] if 'patches' in call['args'] else None
                if isinstance(patches, list) and len(patches) > 0 and isinstance(patches[0], dict) and 'value' in patches[0]:
                    value = patches[0]['value']
                else:
                    value = "No patches available or invalid structure"

                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': value
                })

            elif call['name'] == schema_name:
                # Assume 'args' exists for new schema additions
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)

# Update Memory Tool
class UpdateMemory(TypedDict):
  """ Decision On What Memory Type To Update """
  update_type: Literal["orders"]

class Order(BaseModel):
  """ This is the model for Product which user will order """
  object_name: str
  quantity: int = Field(default=1)
  shipping_address: str

# Initialize the spy for visibility into the tool calls made by Trustcall
spy = Spy()

# Create the Trustcall extractor for updating the user profile
order_extractor = create_extractor(
    model,
    tools=[Order],
    tool_choice="Order",
)

MODEL_SYSTEM_MESSAGE = """
You Are A Helpful Customer Service Assistant

You have two tasks:
- You have to check the product user want to order and if we have the product then you have Update And Delete User Order in order_db
- To adive the user about products and tell him about the products

We have following Products:
<products>
{products}
</products>

These are the orders that user made(This can also be empty):
<orders>
{orders}
</orders>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below.

2. Decide whether any of the your long-term memory should be updated:
- If user wants to order a product the update the order by calling UpdateMemory tool with type `orders`. Always Update the shipping address in the memory.

3. Tell the user that you have updated your memory, if appropriate:
- Tell the user them when you update the order or delete the order

4. User Tell Him Somethings about himself them provide polite reponses
"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

products = [
    {
      "id": 1,
      "name": "Men's Casual Shirt",
      "category": "Men's Wear",
      "price": 25.99,
      "size": ["S", "M", "L", "XL"],
      "color": ["Blue", "Black", "White"],
      "brand": "Urban Styles",
      "stock": 50
    },
    {
      "id": 2,
      "name": "Women's Summer Dress",
      "category": "Women's Wear",
      "price": 35.49,
      "size": ["XS", "S", "M", "L"],
      "color": ["Red", "Yellow", "Floral Print"],
      "brand": "Chic Wear",
      "stock": 30
    },
    {
      "id": 3,
      "name": "Kids' Hoodie",
      "category": "Kids' Wear",
      "price": 19.99,
      "size": ["2-3Y", "4-5Y", "6-7Y"],
      "color": ["Pink", "Green", "Grey"],
      "brand": "MiniCo",
      "stock": 40
    },
    {
      "id": 4,
      "name": "Unisex Sneakers",
      "category": "Footwear",
      "price": 45.99,
      "size": [6, 7, 8, 9, 10, 11],
      "color": ["Black", "White", "Navy Blue"],
      "brand": "StepUp",
      "stock": 60
    },
    {
      "id": 5,
      "name": "Leather Wallet",
      "category": "Accessories",
      "price": 15.99,
      "color": ["Brown", "Black"],
      "brand": "Classic Touch",
      "stock": 100
    },
    {
      "id": 6,
      "name": "Men's Formal Trousers",
      "category": "Men's Wear",
      "price": 40.00,
      "size": ["30", "32", "34", "36", "38"],
      "color": ["Grey", "Black", "Navy"],
      "brand": "Elegant Threads",
      "stock": 20
    },
    {
      "id": 7,
      "name": "Women's Winter Coat",
      "category": "Women's Wear",
      "price": 120.00,
      "size": ["S", "M", "L", "XL"],
      "color": ["Black", "Beige", "Maroon"],
      "brand": "WinterLux",
      "stock": 10
    },
    {
      "id": 8,
      "name": "Sports Cap",
      "category": "Accessories",
      "price": 10.99,
      "color": ["Red", "Blue", "Black"],
      "brand": "Sporty",
      "stock": 75
    },
    {
      "id": 9,
      "name": "Women's Running Shoes",
      "category": "Footwear",
      "price": 55.99,
      "size": [5, 6, 7, 8, 9],
      "color": ["Pink", "Grey", "Black"],
      "brand": "ActiveFit",
      "stock": 25
    },
    {
      "id": 10,
      "name": "Kids' Pajama Set",
      "category": "Kids' Wear",
      "price": 22.50,
      "size": ["2-3Y", "4-5Y", "6-7Y", "8-9Y"],
      "color": ["Yellow", "Blue", "Green"],
      "brand": "Comfort Kids",
      "stock": 35
    }
]

def customer_service(state: MessagesState, config: RunnableConfig, store: BaseStore):
  """ Customer Service Assistant """

  configurable = Configuration.from_runnable_config(config)
  user_id = configurable.user_id

  # Retrieve profile memory from the store
  namespace = ("orders", user_id)
  memories = store.search(namespace)
  if memories:
      orders = memories[0].value
  else:
      orders = None

  system_msg = MODEL_SYSTEM_MESSAGE.format(products=products, orders=orders)

  # Respond using memory as well as the chat history
  response = model.bind_tools([UpdateMemory]).invoke([SystemMessage(content=system_msg)]+state["messages"])

  return {"messages": [response]}

def update_orders(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and update the memory collection."""

    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("orders", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Order"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Invoke the extractor
    result = order_extractor.invoke({"messages": updated_messages,
                                    "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )

    # Respond to the tool call made in task_mAIstro, confirming the update
    tool_calls = state['messages'][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    orders_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": orders_update_msg, "tool_call_id":tool_calls[0]['id']}]}

def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_orders"]: # type: ignore
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state['messages'][-1]

    print("Router", message)

    # If there are no tool calls, end the workflow
    if len(message.tool_calls) == 0:
        return END

    tool_call = message.tool_calls[0]
    update_type = tool_call['args'].get('update_type')

    print(update_type)

    if update_type == "orders":
        return "update_orders"  # Ensure this matches the node name in the graph
    else:
        raise ValueError("Invalid update_type in tool call arguments.")

# Create the graph + all nodes
builder = StateGraph(MessagesState, config_schema=Configuration)

# Define the flow of the memory extraction process
builder.add_node(customer_service)
builder.add_node(update_orders)
builder.add_edge(START, "customer_service")
builder.add_conditional_edges("customer_service", route_message)
builder.add_edge("update_orders", "customer_service")

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# We compile the graph with the checkpointer and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)