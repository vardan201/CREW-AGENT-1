#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from crew import BoardPanelCrew
from models import StartupInput

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def prepare_inputs(startup_data: StartupInput) -> dict:
    """Prepare inputs for all agents from startup data."""
    
    # Marketing inputs
    marketing_channels = ", ".join(startup_data.marketing_growth.current_marketing_channels) if startup_data.marketing_growth.current_marketing_channels else "None specified"
    
    # Tech inputs
    current_features = ", ".join(startup_data.product_technology.current_features) if startup_data.product_technology.current_features else "None specified"
    tech_stack = ", ".join(startup_data.product_technology.tech_stack) if startup_data.product_technology.tech_stack else "None specified"
    
    # Org inputs
    founder_roles = ", ".join(startup_data.team_organization.founder_roles) if startup_data.team_organization.founder_roles else "None specified"
    
    # Competition inputs
    competitors = ", ".join(startup_data.competition_market.known_competitors) if startup_data.competition_market.known_competitors else "None specified"
    
    return {
        # Marketing inputs
        "marketing_channels": marketing_channels,
        "monthly_users": str(startup_data.marketing_growth.monthly_users),
        "cac": startup_data.marketing_growth.customer_acquisition_cost or "Not tracked",
        "retention_strategy": startup_data.marketing_growth.retention_strategy or "No formal strategy",
        "growth_problems": startup_data.marketing_growth.growth_problems or "None specified",
        
        # Tech inputs
        "product_type": startup_data.product_technology.product_type,
        "current_features": current_features,
        "tech_stack": tech_stack,
        "data_strategy": startup_data.product_technology.data_strategy,
        "ai_usage": startup_data.product_technology.ai_usage,
        "tech_challenges": startup_data.product_technology.tech_challenges or "None specified",
        
        # Org inputs
        "team_size": str(startup_data.team_organization.team_size),
        "founder_roles": founder_roles,
        "hiring_plan": startup_data.team_organization.hiring_plan_next_3_months or "No formal plan",
        "org_challenges": startup_data.team_organization.org_challenges or "None specified",
        
        # Competition inputs
        "competitors": competitors,
        "unique_advantage": startup_data.competition_market.unique_advantage or "Not clearly defined",
        "pricing_model": startup_data.competition_market.pricing_model or "Not defined",
        "market_risks": startup_data.competition_market.market_risks or "None identified",
        
        # Finance inputs
        "monthly_burn": startup_data.finance_runway.monthly_burn or "Not tracked",
        "current_revenue": startup_data.finance_runway.current_revenue or "$0",
        "funding_status": startup_data.finance_runway.funding_status,
        "runway_months": startup_data.finance_runway.runway_months or "Unknown",
        "financial_concerns": startup_data.finance_runway.financial_concerns or "None specified",
    }


def run(startup_data: StartupInput):
    """Run the crew with startup data."""
    inputs = prepare_inputs(startup_data)
    
    try:
        result = BoardPanelCrew().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def run_example():
    """Run with example data for testing."""
    from models import (
        ProductTechnology,
        MarketingGrowth,
        TeamOrganization,
        CompetitionMarket,
        FinanceRunway,
        StartupInput
    )
    
    example_data = StartupInput(
        product_technology=ProductTechnology(
            product_type="SaaS",
            current_features=["User Authentication", "Dashboard Analytics", "API Integration"],
            tech_stack=["React", "Node.js", "PostgreSQL", "AWS"],
            data_strategy="User Data",
            ai_usage="Planned",
            tech_challenges="Scaling database queries, implementing real-time features"
        ),
        marketing_growth=MarketingGrowth(
            current_marketing_channels=["Content Marketing", "LinkedIn", "Product Hunt"],
            monthly_users=5000,
            customer_acquisition_cost="$85",
            retention_strategy="Email onboarding sequence, in-app tutorials",
            growth_problems="High churn rate in first 30 days, low organic traffic"
        ),
        team_organization=TeamOrganization(
            team_size=8,
            founder_roles=["CEO", "CTO", "CPO"],
            hiring_plan_next_3_months="1 Senior Engineer, 1 Marketing Manager, 1 Sales Rep",
            org_challenges="Remote team coordination, lack of clear processes"
        ),
        competition_market=CompetitionMarket(
            known_competitors=["Competitor A", "Competitor B", "Open Source Solution"],
            unique_advantage="AI-powered insights and automation",
            pricing_model="Freemium with $49/month Pro plan",
            market_risks="Large incumbents entering the space, economic downturn"
        ),
        finance_runway=FinanceRunway(
            monthly_burn="$75,000",
            current_revenue="$12,000 MRR",
            funding_status="Seed",
            runway_months="14",
            financial_concerns="Need to improve unit economics before Series A"
        )
    )
    
    print("Running Board Panel Analysis with Example Data...")
    print("=" * 80)
    result = run(example_data)
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    return result


if __name__ == "__main__":
    run_example()
