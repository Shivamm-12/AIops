import pandas as pd
import numpy as np

team_sizes = ['Small', 'Medium', 'Large']
release_frequencies = ['Daily', 'Weekly', 'Monthly']
compliance_levels = ['High', 'Medium', 'Low']
scalability_options = ['Auto-scaling', 'Manual', 'None']
project_sizes = ['Small', 'Medium', 'Large']
project_lengths = ['Short', 'Medium', 'Long']
security_levels = ['Strict', 'Moderate', 'None']


def recommend_tools(team_size, release_frequency, compliance, scalability, project_size, project_length, security_level):
    tools = []

    # CI/CD Tool recommendations
    if team_size == 'Small':
        tools.append('Jenkins')
    elif team_size == 'Medium':
        tools.append('CircleCI')
    else:
        tools.append('Travis CI')

    # Version Control recommendations
    if release_frequency == 'Daily':
        tools.append('GitHub')
    elif release_frequency == 'Weekly':
        tools.append('GitLab')
    else:
        tools.append('Bitbucket')

    # Container Tool
    tools.append('Docker')

    # Monitoring Tool recommendations based on Compliance
    if compliance == 'High':
        tools.append('Prometheus')
    elif compliance == 'Medium':
        tools.append('Grafana')
    else:
        tools.append('ELK Stack')

    # Orchestration Tool recommendations
    if scalability == 'Auto-scaling':
        tools.append('Kubernetes')
    elif scalability == 'Manual':
        tools.append('Docker Swarm')
    else:
        tools.append('None')

    # Security Tool recommendations based on Security Level
    if security_level == 'Strict':
        tools.append('HashiCorp Vault')
    elif security_level == 'Moderate':
        tools.append('SonarQube')
    else:
        tools.append('OWASP ZAP')

    # IaC Tool recommendations
    if project_length == 'Short':
        tools.append('Terraform')
    else:
        tools.append('Ansible')

    # Artifact Management recommendations
    if project_size == 'Large':
        tools.append('JFrog Artifactory')
    else:
        tools.append('Nexus Repository')

    return ', '.join(tools)

# Generate dataset
data = []
for i in range(1, 5000):
    team_size = np.random.choice(team_sizes)
    release_frequency = np.random.choice(release_frequencies)
    compliance = np.random.choice(compliance_levels)
    scalability = np.random.choice(scalability_options)
    project_size = np.random.choice(project_sizes)
    project_length = np.random.choice(project_lengths)
    security_level = np.random.choice(security_levels)  

    recommended_tools = recommend_tools(team_size, release_frequency, compliance, scalability, project_size, project_length, security_level)
    
    data.append([
        i, team_size, release_frequency, compliance, scalability, project_size, project_length, security_level, recommended_tools
    ])

df = pd.DataFrame(data, columns=['S.No', 'Team Size', 'Release Frequency', 'Compliance', 'Scalability', 'Project Size', 'Project Length', 'Security Level', 'Recommended Tools'])


df.to_csv('devops_tool_recommendations.csv', index=False)

#print("Dataset generated and saved to devops_tool_recommendations.csv.")
