"""
Example: Customer Support Agent with Skills

A realistic customer support agent that uses skills for different
support domains.  Each skill packages domain-specific instructions
and tools together, so the agent knows HOW to use each tool
(via instructions) and WHAT tools are available (via the skill's
tool list).

Skills:
- **order-management** — Track orders, process returns, check inventory
- **billing** — Check balances, process payments, apply discounts

The agent also uses SkillDiscoveryToolset so the LLM can introspect
what skills and tools are available — useful for complex support
scenarios.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.llms import LLMConfig
from troopai.adk.run import RunConfig, Runner
from troopai.adk.skills import (
    Skill,
    SkillActivation,
    SkillDiscoveryToolset,
    SkillGovernance,
)
from troopai.adk.tools import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Order Management Tools
# =============================================================================

ORDERS_DB = {
    "ORD-001": {"status": "shipped", "item": "Wireless Headphones", "total": 79.99, "tracking": "1Z999AA1"},
    "ORD-002": {"status": "processing", "item": "USB-C Hub", "total": 34.99, "tracking": None},
    "ORD-003": {"status": "delivered", "item": "Laptop Stand", "total": 49.99, "tracking": "1Z999AA3"},
}


@function_tool(
    name="lookup_order",
    description="Look up an order by order ID. Returns status, item, total, and tracking number.",
)
def lookup_order(order_id: str) -> str:
    """Look up order details."""
    order = ORDERS_DB.get(order_id.upper())
    if order is None:
        return f"Order {order_id} not found. Valid order IDs: {', '.join(ORDERS_DB)}"
    tracking = order["tracking"] or "Not yet assigned"
    return (
        f"Order {order_id}: {order['item']} | "
        f"Status: {order['status']} | "
        f"Total: ${order['total']:.2f} | "
        f"Tracking: {tracking}"
    )


@function_tool(
    name="initiate_return",
    description="Initiate a return for a delivered order. Returns a return authorization number.",
)
def initiate_return(order_id: str, reason: str) -> str:
    """Start a return process."""
    order = ORDERS_DB.get(order_id.upper())
    if order is None:
        return f"Order {order_id} not found"
    if order["status"] != "delivered":
        return f"Cannot return order {order_id} — status is '{order['status']}' (must be 'delivered')"
    return f"Return initiated for {order_id}. Return Auth: RMA-{order_id[-3:]}. Reason: {reason}. Ship back within 14 days."


@function_tool(
    name="check_inventory",
    description="Check if a product is in stock.",
)
def check_inventory(product_name: str) -> str:
    """Check product availability."""
    inventory = {
        "wireless headphones": 42,
        "usb-c hub": 0,
        "laptop stand": 15,
        "mechanical keyboard": 28,
    }
    qty = inventory.get(product_name.lower())
    if qty is None:
        return f"Product '{product_name}' not found in catalog"
    if qty == 0:
        return f"'{product_name}' is currently OUT OF STOCK. Expected restock: 5-7 business days."
    return f"'{product_name}' is IN STOCK ({qty} units available)"


# =============================================================================
# Billing Tools
# =============================================================================

ACCOUNTS_DB = {
    "ACCT-100": {"name": "Alice", "balance": 150.00, "credit": 25.00},
    "ACCT-200": {"name": "Bob", "balance": 0.00, "credit": 0.00},
}


@function_tool(
    name="check_balance",
    description="Check an account balance and available credit.",
)
def check_balance(account_id: str) -> str:
    """Check account balance."""
    account = ACCOUNTS_DB.get(account_id.upper())
    if account is None:
        return f"Account {account_id} not found"
    return (
        f"Account {account_id} ({account['name']}): "
        f"Balance due: ${account['balance']:.2f} | "
        f"Store credit: ${account['credit']:.2f}"
    )


@function_tool(
    name="apply_discount",
    description="Apply a percentage discount to an account's balance.",
)
def apply_discount(account_id: str, discount_percent: float) -> str:
    """Apply a discount."""
    account = ACCOUNTS_DB.get(account_id.upper())
    if account is None:
        return f"Account {account_id} not found"
    if discount_percent > 20:
        return f"Cannot apply {discount_percent}% discount. Maximum is 20%."
    original = account["balance"]
    savings = original * (discount_percent / 100)
    new_balance = original - savings
    return (
        f"Applied {discount_percent}% discount to {account_id}. "
        f"Was: ${original:.2f} → Now: ${new_balance:.2f} (saved ${savings:.2f})"
    )


# =============================================================================
# Skill Definitions
# =============================================================================

order_skill = Skill(
    name="order-management",
    description="Order tracking, returns, and inventory management",
    instructions="""When handling order inquiries:
1. Always look up the order first with lookup_order
2. If the customer wants to return, check the order status — only 'delivered' orders can be returned
3. For out-of-stock items, check_inventory and provide the restock estimate
4. Be empathetic and solution-oriented — suggest alternatives if an item is unavailable
5. Always confirm the order details with the customer before processing returns""",
    tools=[lookup_order, initiate_return, check_inventory],
    governance=SkillGovernance(timeout=10.0, max_result_tokens=500),
)

billing_skill = Skill(
    name="billing",
    description="Account balances, payments, and discounts",
    instructions="""When handling billing inquiries:
1. Always check the account balance first
2. Only apply discounts when the customer has a legitimate reason (loyalty, issue resolution)
3. Maximum discount is 20% — explain this limit if the customer asks for more
4. Mention available store credit when checking balances
5. Be transparent about all charges and adjustments""",
    tools=[check_balance, apply_discount],
    governance=SkillGovernance(timeout=10.0),
)


# =============================================================================
# Agent with Discovery Toolset
# =============================================================================

skills = [order_skill, billing_skill]
discovery = SkillDiscoveryToolset(skills=skills)

support_agent = Agent(
    name="Customer Support",
    system_prompt=(
        "You are a friendly and efficient customer support agent for TechShop, "
        "an online electronics retailer. Help customers with their orders, "
        "returns, billing, and product inquiries. Be concise but thorough."
    ),
    skills=skills,
    skill_activation=SkillActivation.EAGER,
    tools=[*discovery.tools()],  # LLM can introspect available skills
    llm_config=LLMConfig(temperature=0.2),
)


# =============================================================================
# Run Support Scenarios
# =============================================================================


async def run_order_scenario() -> None:
    """Customer asking about an order and initiating a return."""
    logger.info("=" * 60)
    logger.info("Scenario 1: Order inquiry + return")
    logger.info("=" * 60)

    result = await Runner.arun(
        support_agent,
        "Hi, I'd like to check on my order ORD-003. If it's been delivered, "
        "I'd like to return it — the stand doesn't fit my desk.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Agent response:\n%s", result.final_output)
    logger.info("Tokens used: %s", result.context.usage.total_tokens)


async def run_billing_scenario() -> None:
    """Customer asking about balance and requesting a discount."""
    logger.info("=" * 60)
    logger.info("Scenario 2: Balance check + discount request")
    logger.info("=" * 60)

    result = await Runner.arun(
        support_agent,
        "Can you check my account ACCT-100? I've been a loyal customer for "
        "3 years and I'd appreciate a 10% loyalty discount on my balance.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Agent response:\n%s", result.final_output)
    logger.info("Tokens used: %s", result.context.usage.total_tokens)


async def run_cross_skill_scenario() -> None:
    """Customer query that spans both skills."""
    logger.info("=" * 60)
    logger.info("Scenario 3: Cross-skill query (order + billing)")
    logger.info("=" * 60)

    result = await Runner.arun(
        support_agent,
        "I want to reorder the USB-C Hub from ORD-002. Is it in stock? "
        "Also, check if I have any credit on account ACCT-100 I could use.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Agent response:\n%s", result.final_output)
    logger.info("Tokens used: %s", result.context.usage.total_tokens)


async def main() -> None:
    await run_order_scenario()
    logger.info("")
    await run_billing_scenario()
    logger.info("")
    await run_cross_skill_scenario()


if __name__ == "__main__":
    asyncio.run(main())
