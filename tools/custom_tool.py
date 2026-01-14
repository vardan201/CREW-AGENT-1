from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MarketResearchInput(BaseModel):
    """Input for market research tool."""
    query: str = Field(..., description="The market research query or topic to investigate")
    industry: str = Field(..., description="The industry or sector to focus on")


class MarketResearchTool(BaseTool):
    name: str = "Market Research Tool"
    description: str = (
        "Performs market research and analysis on specific industries, competitors, "
        "and market trends. Provides insights on market size, growth rates, and competitive dynamics."
    )
    args_schema: Type[BaseModel] = MarketResearchInput

    def _run(self, query: str, industry: str) -> str:
        # Simulated market research - in production, this could call external APIs
        return f"""
Market Research Results for {industry}:

Query: {query}

Key Findings:
- Market is showing strong growth potential
- Competitive landscape is moderating intense
- Customer acquisition trends favor digital channels
- Emerging technologies are creating new opportunities
- Regulatory environment is stable with minor changes expected

Recommendation: Focus on differentiation and customer experience to stand out in the market.
"""


class FinancialBenchmarkInput(BaseModel):
    """Input for financial benchmark tool."""
    company_stage: str = Field(..., description="The company stage (Seed, Series A, etc.)")
    industry: str = Field(..., description="The industry sector")


class FinancialBenchmarkTool(BaseTool):
    name: str = "Financial Benchmark Tool"
    description: str = (
        "Provides financial benchmarking data for startups based on stage and industry. "
        "Returns metrics like typical burn rates, revenue multiples, and growth rates."
    )
    args_schema: Type[BaseModel] = FinancialBenchmarkInput

    def _run(self, company_stage: str, industry: str) -> str:
        # Simulated benchmarking data
        benchmarks = {
            "Seed": {
                "typical_burn": "$50K-150K/month",
                "runway_months": "12-18 months",
                "growth_rate": "15-25% MoM"
            },
            "Series A": {
                "typical_burn": "$200K-500K/month",
                "runway_months": "18-24 months",
                "growth_rate": "10-20% MoM"
            }
        }
        
        stage_data = benchmarks.get(company_stage, benchmarks["Seed"])
        
        return f"""
Financial Benchmarks for {company_stage} in {industry}:

Typical Monthly Burn: {stage_data['typical_burn']}
Expected Runway: {stage_data['runway_months']}
Target Growth Rate: {stage_data['growth_rate']}

Industry Context:
- Companies at this stage typically focus on product-market fit
- Unit economics become critical for next funding round
- CAC payback period should be under 12 months
- Gross margins should exceed 70% for SaaS
"""


class TechStackAnalysisInput(BaseModel):
    """Input for tech stack analysis tool."""
    tech_stack: str = Field(..., description="Comma-separated list of technologies")
    product_type: str = Field(..., description="Type of product (Web, Mobile, SaaS, etc.)")


class TechStackAnalysisTool(BaseTool):
    name: str = "Tech Stack Analysis Tool"
    description: str = (
        "Analyzes technology stack choices for scalability, security, and maintainability. "
        "Provides recommendations on tech stack improvements and identifies potential issues."
    )
    args_schema: Type[BaseModel] = TechStackAnalysisInput

    def _run(self, tech_stack: str, product_type: str) -> str:
        return f"""
Tech Stack Analysis for {product_type}:

Current Stack: {tech_stack}

Analysis:
- Scalability: Stack shows good horizontal scaling potential
- Security: Modern security practices should be implemented
- Maintainability: Popular technologies with strong community support
- Performance: Optimize database queries and implement caching
- Cost: Cloud costs may increase with scale - consider optimization

Recommendations:
1. Implement comprehensive monitoring and logging
2. Add CI/CD pipeline if not present
3. Consider containerization for easier deployment
4. Implement proper error tracking and monitoring
5. Set up automated testing infrastructure
"""


class HiringStrategyInput(BaseModel):
    """Input for hiring strategy tool."""
    team_size: int = Field(..., description="Current team size")
    roles_needed: str = Field(..., description="Roles that need to be filled")


class HiringStrategyTool(BaseTool):
    name: str = "Hiring Strategy Tool"
    description: str = (
        "Provides strategic hiring recommendations based on company stage, team size, "
        "and role requirements. Suggests hiring priorities and compensation benchmarks."
    )
    args_schema: Type[BaseModel] = HiringStrategyInput

    def _run(self, team_size: int, roles_needed: str) -> str:
        return f"""
Hiring Strategy Analysis:

Current Team Size: {team_size}
Roles Needed: {roles_needed}

Strategic Recommendations:

Priority Hiring Order:
1. Critical revenue-generating roles first
2. Technical roles to support product development
3. Operational roles to scale efficiently

Hiring Best Practices:
- Define clear role requirements and success metrics
- Implement structured interview process
- Check for culture fit alongside skills
- Provide competitive but sustainable compensation
- Consider contract-to-hire for specialized roles

Compensation Guidance:
- Equity: 0.1%-1.0% for early employees depending on role and seniority
- Salary: Market rate or slightly below with equity compensation
- Benefits: Health insurance, flexible work arrangements

Timeline:
- Plan for 2-3 months per critical hire
- Build pipeline continuously, not just when desperate
"""


class CompetitorAnalysisInput(BaseModel):
    """Input for competitor analysis tool."""
    competitors: str = Field(..., description="Comma-separated list of competitors")
    industry: str = Field(..., description="Industry sector")


class CompetitorAnalysisTool(BaseTool):
    name: str = "Competitor Analysis Tool"
    description: str = (
        "Performs competitive analysis by examining competitor strategies, strengths, "
        "weaknesses, and market positioning. Identifies opportunities for differentiation."
    )
    args_schema: Type[BaseModel] = CompetitorAnalysisInput

    def _run(self, competitors: str, industry: str) -> str:
        return f"""
Competitive Analysis for {industry}:

Competitors Analyzed: {competitors}

Competitive Landscape Overview:
- Market is moderately competitive with 5-10 major players
- Differentiation opportunities exist in customer experience
- Price competition is present but quality matters more
- Technology innovation is a key differentiator

SWOT Analysis Framework:
Opportunities:
- Underserved customer segments exist
- Emerging technologies enable new features
- Market is growing faster than competition can scale

Threats:
- Well-funded competitors may copy innovations
- Market consolidation could change dynamics
- New entrants with novel approaches

Strategic Positioning Recommendations:
1. Focus on a specific niche or vertical
2. Build strong customer relationships and community
3. Invest in unique technology capabilities
4. Emphasize superior customer support
5. Create high switching costs through integrations

Differentiation Strategy:
- Product: Unique features or superior user experience
- Price: Value-based pricing with clear ROI
- Service: Exceptional customer success and support
- Brand: Strong brand identity and market presence
"""
