-- Supabase/Postgres migration for dynamic intent routing registry.
-- Run this once in the Supabase SQL editor for your project.

create extension if not exists pgcrypto;

create table if not exists public.rules_registry (
    id uuid primary key default gen_random_uuid(),
    intent text not null,
    enabled boolean not null default true,
    priority integer not null default 100,
    patterns text[] not null,
    builder text,
    sql_template text,
    description text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint rules_registry_builder_or_template_chk
        check (
            nullif(trim(coalesce(builder, '')), '') is not null
            or nullif(trim(coalesce(sql_template, '')), '') is not null
        ),
    constraint rules_registry_patterns_nonempty_chk
        check (coalesce(array_length(patterns, 1), 0) > 0)
);

create index if not exists rules_registry_priority_idx
    on public.rules_registry (enabled, priority);

create index if not exists rules_registry_intent_idx
    on public.rules_registry (intent);

create or replace function public.rules_registry_set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_rules_registry_set_updated_at on public.rules_registry;
create trigger trg_rules_registry_set_updated_at
before update on public.rules_registry
for each row
execute function public.rules_registry_set_updated_at();
