export type Provider = 'google' | 'github';
export type ProjectStatus = 'active' | 'disabled';

export interface ManagedUser {
  id: string;
  email: string | null;
  name: string | null;
  avatarUrl: string | null;
}

export interface ManagedProject {
  projectId: string;
  name: string;
  homepageUrl: string;
  description: string;
  avatarUrl: string | null;
  allowedRedirectUris: string[];
  allowedOrigins: string[];
  enabledProviders: Provider[];
  status: ProjectStatus;
}

export class ManagementError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

const projectsKey = 'nexuss-preview-projects';
const sessionKey = 'nexuss-preview-session';

function readProjects(): ManagedProject[] {
  try {
    return JSON.parse(localStorage.getItem(projectsKey) ?? '[]') as ManagedProject[];
  } catch {
    return [];
  }
}

function requireSession(): void {
  if (localStorage.getItem(sessionKey) !== 'active') throw new ManagementError(401, 'Sign-in required');
}

export async function getCurrentUser(): Promise<ManagedUser | null> {
  requireSession();
  return { id: 'preview-user', email: 'you@example.com', name: 'Signed-in user', avatarUrl: null };
}

export async function listManagedProjects(): Promise<ManagedProject[]> {
  requireSession();
  return readProjects();
}

export async function createManagedProject(project: ManagedProject): Promise<ManagedProject> {
  requireSession();
  const next = [...readProjects(), project];
  localStorage.setItem(projectsKey, JSON.stringify(next));
  return project;
}

export async function updateManagedProject(projectId: string, updates: Partial<Omit<ManagedProject, 'projectId'>>): Promise<ManagedProject> {
  requireSession();
  const current = readProjects();
  const existing = current.find((project) => project.projectId === projectId);
  if (!existing) throw new ManagementError(404, 'Project not found');
  const updated = { ...existing, ...updates };
  localStorage.setItem(projectsKey, JSON.stringify(current.map((project) => project.projectId === projectId ? updated : project)));
  return updated;
}

export function beginDashboardSignIn(_provider: Provider): void {
  localStorage.setItem(sessionKey, 'active');
  window.location.reload();
}
